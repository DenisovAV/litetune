"""Draw `web/images/og-image.png`, the 1200x630 card link previews show.

Run from `website/` when the wording or the palette changes, and commit the PNG:

    uv run --with pillow==12.3.0 python tool/og_image.py <font-dir>

<font-dir> holds `IBMPlexMono-Light.ttf`, `IBMPlexMono-SemiBold.ttf` and
`IBMPlexSans-Light.ttf` -- the last is the variable `IBMPlexSans[wdth,wght].ttf`
from github.com/google/fonts, renamed. They are the faces the page loads from
Google Fonts and are not in this repository. The colours are the ones
`lib/theme/brand.dart` declares.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630
BG = "#0B0B0A"
INK = "#F5F5F3"
BODY = "#C9C6BF"
MUTED = "#9A968E"
LINE = "#2A2A27"
MARGIN = 96

# `verify --backend` takes cpu or gpu (evaluate.CANDIDATE_BACKENDS); an NPU file
# comes out of `convert --npu`, and `verify` does not run one.
TAGLINE = (
    "Fine-tune small models, convert to LiteRT-LM,",
    "verify against the float model on CPU and GPU.",
)
FOOTER = "litetune.dev"


def draw(font_dir: Path, out: Path) -> None:
    lite = ImageFont.truetype(str(font_dir / "IBMPlexMono-Light.ttf"), 132)
    tune = ImageFont.truetype(str(font_dir / "IBMPlexMono-SemiBold.ttf"), 132)
    sans = ImageFont.truetype(str(font_dir / "IBMPlexSans-Light.ttf"), 44)
    sans.set_variation_by_name("Light")
    mono = ImageFont.truetype(str(font_dir / "IBMPlexMono-Light.ttf"), 30)

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    pen = ImageDraw.Draw(image)

    # The wordmark as the nav bar draws it: Plex Mono, "lite" light and "tune"
    # semibold, tracked at -0.04em.
    x: float = MARGIN
    for text, font in (("lite", lite), ("tune", tune)):
        for char in text:
            pen.text((x, 120), char, font=font, fill=INK)
            x += font.getlength(char) - 0.04 * font.size

    y = 330
    for line in TAGLINE:
        pen.text((MARGIN, y), line, font=sans, fill=BODY)
        y += 62

    pen.line([(MARGIN, 500), (WIDTH - MARGIN, 500)], fill=LINE, width=2)
    pen.text((MARGIN, 528), FOOTER, font=mono, fill=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    draw(Path(sys.argv[1]), Path(__file__).resolve().parent.parent / "web/images/og-image.png")
