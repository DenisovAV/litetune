/// What the site says about the package, read from the repository at build time.
///
/// The site is pre-rendered: `jaspr build` runs the server entrypoint once, on
/// the machine doing the build, and writes HTML. So a version number or a
/// changelog written into a Dart constant would be a second copy of something
/// the package already states, and the copy that goes stale is the one visitors
/// read. These are the package's own files instead.
///
/// Server-only. `dart:io` is not available to the client build, which is why
/// the components that show these values take them as constructor arguments
/// rather than importing this file.
library;

import 'dart:io';

/// The repository root, found from the working directory `jaspr build` runs in.
///
/// Both callers -- `deploy.sh` and the CI build -- run it from `website/`, so
/// the root is one level up. Found by looking for the file rather than assumed,
/// so a build started somewhere else fails naming what it looked for instead of
/// rendering a page with no version on it.
Directory _repositoryRoot() {
  for (final candidate in [Directory.current.parent, Directory.current]) {
    if (File('${candidate.path}/src/litetune/_version.py').existsSync()) {
      return candidate;
    }
  }
  throw StateError(
    'cannot find src/litetune/_version.py from ${Directory.current.path}; '
    'the site is built from website/ inside the litetune repository',
  );
}

final Directory _root = _repositoryRoot();

/// `__version__` from `src/litetune/_version.py`, the number PyPI publishes.
///
/// Matched as a whole line, not as the first quoted string in the file: the
/// module docstring above the assignment contains quoted text of its own, and
/// `publish.yml` records a guard that once read a docstring as the version.
///
/// Exactly one such line, as `check-build.sh` and the deploy job require.
/// With two, this would render the first while Python imports the last, and
/// the page and the package would name different versions.
final String packageVersion = () {
  final source = File('${_root.path}/src/litetune/_version.py').readAsStringSync();
  final matches = RegExp(
    r'^__version__ = "([^"]+)"$',
    multiLine: true,
  ).allMatches(source).toList();
  final assignments = RegExp(r'^__version__ = ', multiLine: true).allMatches(source).length;
  if (matches.length != 1 || assignments != 1) {
    throw StateError(
      'src/litetune/_version.py must have exactly one `__version__ = "..."` line; '
      'found $assignments assignment(s), ${matches.length} readable',
    );
  }
  return matches.single.group(1)!;
}();

/// `CHANGELOG.md` at the repository root, as written.
final String changelogMarkdown = File(
  '${_root.path}/CHANGELOG.md',
).readAsStringSync();
