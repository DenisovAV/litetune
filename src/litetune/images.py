"""The images of an image decision split: checked with Pillow, copied byte for byte.

An image row of `prepare` names a file. Two things are done to it here and
nothing else.

**It is opened with the Pillow `tune` trains with.** Pillow is not a host
dependency; it is in `envs.EMBED_TRAIN` (pinned with the recipe). So the check
runs there, as a script, the way `prepare.HuggingFaceTokenCounter` runs its
tokenizer: the parent never imports it. The script reports what it found per
file -- format, size, frame count, the sha256 of the bytes it decoded, or the
decoder's error -- and the refusals are decided in `prepare`, where they name
the row.

**It is copied, never re-encoded.** The split is self-contained:
`images/<sha256>.<ext>` beside the JSONL, the bytes the user gave. The
fine-tuned model is sensitive to compression (the spec, 2026-11-04-eg2-image-tune:
one crop as PNG and as JPEG embedded at cosine 0.918 on the fine-tune), so a
split that re-encoded would train on a format the application never feeds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from litetune import envs
from litetune.exits import read_returncode

logger = logging.getLogger(__name__)

# The directory of a split that holds its images, and the prefix a split row's
# `image` carries.
IMAGES_DIR = "images"

# Opening a few thousand small files is metadata work, as counting tokens is
# (`prepare.TOKEN_COUNT_TIMEOUT_S`).
IMAGE_CHECK_TIMEOUT_S = 600

_CHUNK_BYTES = 1 << 20

# Pillow's format names are upper-case words ("PNG", "JPEG", "WEBP"). Anything
# else in an answer is not one, and it would become part of a file name.
_FORMAT = re.compile(r"[A-Z0-9]{2,16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# The one spelling that differs from the format name lower-cased.
_EXTENSIONS = {"jpeg": "jpg"}


class ImageCheckUnavailable(RuntimeError):
    """The images could not be checked. Not a statement about the images."""


@dataclass(frozen=True)
class ImageFacts:
    """What Pillow made of one file: its format and size, or the error that stopped it."""

    sha256: str | None
    format: str | None
    width: int | None
    height: int | None
    frames: int | None
    error: str | None

    @property
    def extension(self) -> str:
        """The split file's extension: the decoded format, lower case, `jpeg` as `jpg`."""
        if self.format is None:
            raise ValueError("an image that did not decode has no extension")
        lower = self.format.lower()
        return _EXTENSIONS.get(lower, lower)


class ImageInspector(Protocol):
    """Opens images the way `tune` will. A Protocol so the tests need no Pillow."""

    def describe(self) -> dict[str, Any]:
        """What checked the images. Travels with the report."""

    def inspect(self, paths: Sequence[Path]) -> list[ImageFacts]:
        """One `ImageFacts` per path, in order. Raises `ImageCheckUnavailable` if it
        could not run."""


# Runs inside envs.EMBED_TRAIN, whose Pillow is the one `tune` decodes with.
# The bytes are read once: the digest and the decode are of the same bytes, so
# a file replaced between the two cannot be hashed as one image and checked as
# another.
IMAGE_CHECK_SCRIPT = r'''
"""Open each image with Pillow. Writes JSON: {"pillow": str, "images": [facts, ...]}."""
import hashlib
import io
import json
import sys
import warnings
from pathlib import Path


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    import PIL
    from PIL import Image

    # A decompression bomb is refused, not warned about: past Pillow's pixel
    # limit it only warns, and the run would go on to decode it.
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    found = [inspect(Image, path) for path in spec["paths"]]
    Path(spec["out"]).write_text(
        json.dumps({"pillow": PIL.__version__, "images": found}), encoding="utf-8"
    )
    return 0


def inspect(Image, path):
    facts = {
        "sha256": None, "format": None, "width": None, "height": None, "frames": None,
        "error": None,
    }
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
        return facts
    facts["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as image:
            facts["format"] = image.format
            facts["width"], facts["height"] = image.size
            facts["frames"] = int(getattr(image, "n_frames", 1))
            image.load()
            image.convert("RGB")
    # Every decoder failure is this file's answer, reported per file: Pillow
    # raises OSError, SyntaxError, ValueError, a DecompressionBombError and
    # more, and one bad file must not hide what the others are.
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


if __name__ == "__main__":
    sys.exit(main())
'''


def facts_from(answer: Mapping[str, Any]) -> ImageFacts:
    """One file's answer from the script, checked to be the shape it writes."""
    sha = answer.get("sha256")
    fmt = answer.get("format")
    error = answer.get("error")
    sizes = [answer.get(key) for key in ("width", "height", "frames")]
    if sha is not None and not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        raise ImageCheckUnavailable(f"the image check answered a sha256 of {sha!r}")
    if error is None:
        if sha is None:
            raise ImageCheckUnavailable("the image check answered no sha256 and no error")
        if not (isinstance(fmt, str) and _FORMAT.fullmatch(fmt)):
            raise ImageCheckUnavailable(f"the image check answered a format of {fmt!r}")
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in sizes):
            raise ImageCheckUnavailable(f"the image check answered sizes {sizes!r}")
    elif not isinstance(error, str):
        raise ImageCheckUnavailable(f"the image check answered an error of {error!r}")
    return ImageFacts(
        sha256=sha,
        format=fmt if error is None else None,
        width=sizes[0] if error is None else None,
        height=sizes[1] if error is None else None,
        frames=sizes[2] if error is None else None,
        error=error,
    )


@dataclass
class PillowImageInspector:
    """The real check: `IMAGE_CHECK_SCRIPT` inside `envs.EMBED_TRAIN`, one subprocess a call."""

    env: envs.StageEnv = envs.EMBED_TRAIN
    timeout_s: int = IMAGE_CHECK_TIMEOUT_S
    auto_provision: bool = True
    # Filled in by `inspect` from the script's answer.
    pillow: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "checker": "pillow",
            "pillow": self.pillow,
            "environment": self.env.name,
            "requirements": envs.shown_requirements(self.env.requirements),
        }

    def inspect(self, paths: Sequence[Path]) -> list[ImageFacts]:
        if self.auto_provision:
            try:
                self.env.provision()
            except (RuntimeError, OSError) as exc:
                logger.exception("could not provision environment %r", self.env.name)
                raise ImageCheckUnavailable(
                    f"environment {self.env.name!r} unavailable: {exc}"
                ) from exc

        with tempfile.TemporaryDirectory(prefix="litetune-images-") as tmp:
            work = Path(tmp)
            script = work / "check_images.py"
            script.write_text(IMAGE_CHECK_SCRIPT, encoding="utf-8")
            out = work / "images.json"
            config = work / "config.json"
            config.write_text(
                json.dumps({"paths": [str(p) for p in paths], "out": str(out)}), encoding="utf-8"
            )
            try:
                proc = self.env.run(["python", str(script), str(config)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                logger.warning("the image check did not answer within %ss", self.timeout_s)
                raise ImageCheckUnavailable(
                    f"the image check did not answer within {self.timeout_s}s"
                ) from exc
            except OSError as exc:
                logger.exception("could not start the image check script")
                raise ImageCheckUnavailable(f"{type(exc).__name__}: {exc}") from exc
            if proc.returncode != 0 or not out.is_file():
                reading = read_returncode(proc.returncode)
                raise ImageCheckUnavailable(
                    f"the image check was {reading.describe('the images')} without an answer: "
                    f"{(proc.stderr or '').strip()[-300:] or 'no stderr'}"
                )
            payload = json.loads(out.read_text(encoding="utf-8"))
        answers = payload.get("images")
        if not isinstance(answers, list) or len(answers) != len(paths):
            raise ImageCheckUnavailable(
                f"the image check answered for {len(answers) if isinstance(answers, list) else 0} "
                f"of {len(paths)} images"
            )
        pillow = payload.get("pillow")
        self.pillow = pillow if isinstance(pillow, str) else None
        if not all(isinstance(a, dict) for a in answers):
            raise ImageCheckUnavailable("the image check answered something other than an object")
        return [facts_from(a) for a in answers]


@dataclass(frozen=True)
class SplitImage:
    """An image row's file: where it is read from, and what Pillow made of it."""

    source: Path
    sha256: str
    format: str
    width: int
    height: int
    extension: str

    @property
    def name(self) -> str:
        """The file's name in the split: content-addressed, so one picture is one file."""
        return f"{self.sha256}.{self.extension}"

    @property
    def relative(self) -> str:
        """The split row's `image`: relative to the split, with `/` on every platform."""
        return f"{IMAGES_DIR}/{self.name}"


class ImageCopyError(ValueError):
    """An image could not be copied into the split as the bytes that were checked."""


def _copy_checked(source: Path, dest: Path, sha256: str) -> None:
    """Copy `source` to `dest`, refusing bytes that are not the ones that were checked."""
    digest = hashlib.sha256()
    with source.open("rb") as reader, dest.open("xb") as writer:
        for chunk in iter(lambda: reader.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
            writer.write(chunk)
    if digest.hexdigest() != sha256:
        raise ImageCopyError(
            f"{source} changed after it was checked: its bytes now hash to {digest.hexdigest()}, "
            f"not {sha256}. Run prepare again once nothing is writing to it"
        )


def _remove_flat(directory: Path) -> None:
    """Remove a directory this module wrote: files only, never a tree."""
    for entry in directory.iterdir():
        entry.unlink()
    directory.rmdir()


def _fresh_dir(parent: Path, prefix: str) -> Path:
    """A new directory under `parent`, made with the umask's mode.

    Not `tempfile.mkdtemp`: its 0700 would become the split's `images` mode on
    the rename, and a split is read by whoever trains on it.
    """
    for _ in range(100):
        candidate = parent / f"{prefix}{secrets.token_hex(8)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"no free name for a {prefix}* directory in {parent}")


def write_images(output_dir: Path, sources: Mapping[str, Path]) -> Path:
    """`output_dir/images`, holding exactly `sources` as `<name>` -> the bytes of its path.

    `sources` maps a file name (`<sha256>.<ext>`) to the file it is copied from.
    The directory is built beside its final place and renamed into it, so a copy
    that fails part way leaves the split's previous images, or none, and never
    a half of them. A previous `images` directory is replaced, not merged: its
    files may belong to rows this split no longer has.
    """
    target = output_dir / IMAGES_DIR
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ImageCopyError(f"{target} exists and is not a directory; move it away")
    if target.is_dir():
        foreign = [e.name for e in target.iterdir() if e.is_dir() or e.is_symlink()]
        if foreign:
            raise ImageCopyError(
                f"{target} holds {', '.join(sorted(foreign)[:5])}, which prepare did not write "
                "(it writes files only); move it away"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = _fresh_dir(output_dir, ".images-")
    try:
        for name, source in sorted(sources.items()):
            _copy_checked(source, staging / name, name.split(".", 1)[0])
    except BaseException:
        _remove_flat(staging)
        raise
    retired: Path | None = None
    if target.is_dir():
        retired = _fresh_dir(output_dir, ".images-old-") / IMAGES_DIR
        target.rename(retired)
    try:
        staging.rename(target)
    except OSError:
        logger.exception("could not move %s into place as %s", staging, target)
        if retired is not None:
            retired.rename(target)
            retired.parent.rmdir()
        _remove_flat(staging)
        raise
    if retired is not None:
        _remove_flat(retired)
        retired.parent.rmdir()
    return target
