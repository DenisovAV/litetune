// The panel's links, checked against the files they point into.
//
// Each card links to a section of `MEASUREMENTS.md` by GitHub's own heading
// anchor. Nothing in a Dart build knows whether that section still exists, and
// a renamed heading leaves a link that lands on the top of a 900-line file
// instead of the run it promised -- silently, on the live site. So the anchors
// are computed from the headings here and compared with the ones the cards
// carry.
//
// The same check covers the Hub links by their shape only: whether
// huggingface.co still serves a repository is not something a unit test can
// answer, and a test that pretends to would be worse than none.
import 'dart:io';

import 'package:litetune_website/landing/sections/why_it_exists.dart';
import 'package:test/test.dart';

/// GitHub's anchor for a markdown heading: lowercased, punctuation dropped,
/// spaces hyphenated.
String anchorFor(String heading) => heading
    .toLowerCase()
    .replaceAll(RegExp(r'[^\w\- ]'), '')
    .replaceAll(' ', '-');

File _measurements() {
  for (final candidate in [Directory.current.parent, Directory.current]) {
    final file = File('${candidate.path}/MEASUREMENTS.md');
    if (file.existsSync()) return file;
  }
  throw StateError(
    'cannot find MEASUREMENTS.md from ${Directory.current.path}',
  );
}

/// The file by heading anchor, each entry running to the next heading of the
/// same level or higher.
///
/// So a `##` entry carries its `###` subsections -- which is what a reader
/// following the link sees, and where most of the detail lives: the
/// FunctionGemma run names its checkpoint in a subsection of "The headline
/// numbers", not under the heading itself -- while a `###` entry exists in
/// its own right, so a card linked to a subsection is looked up rather than
/// reported as a heading that does not exist.
Map<String, String> _sectionsByAnchor(List<String> lines) {
  final sections = <String, StringBuffer>{};
  final open = <int, StringBuffer>{};
  for (final line in lines) {
    final heading = RegExp(r'^(#{2,3}) (.+)$').firstMatch(line);
    if (heading != null) {
      final level = heading.group(1)!.length;
      open.removeWhere((depth, _) => depth >= level);
      open[level] = sections[anchorFor(heading.group(2)!.trim())] =
          StringBuffer();
    }
    for (final buffer in open.values) {
      buffer.writeln(line);
    }
  }
  return {for (final e in sections.entries) e.key: e.value.toString()};
}

File _readme() {
  final file = File('${_measurements().parent.path}/README.md');
  if (!file.existsSync()) {
    throw StateError('cannot find README.md beside MEASUREMENTS.md');
  }
  return file;
}

void main() {
  final sections = _sectionsByAnchor(_measurements().readAsLinesSync());
  final readme = _readme().readAsStringSync();

  test('MEASUREMENTS.md has the sections the cards link to', () {
    expect(sections, isNotEmpty, reason: 'no headings were found at all');
    for (final model in WhyItExists.measured) {
      expect(
        sections.keys,
        contains(model.anchor),
        reason:
            '${model.name} links to MEASUREMENTS.md#${model.anchor}, and no '
            'heading in that file produces this anchor',
      );
    }
  });

  test('each card links to a section that names its own checkpoint', () {
    // What pins a card to a run when it carries no revision: FunctionGemma's
    // does not, so without this its anchor could point at any section that
    // exists. The section has to name the repository the panel shows.
    for (final model in WhyItExists.measured) {
      expect(
        sections[model.anchor],
        contains(model.hubId.split('/').last),
        reason:
            '${model.name} links to #${model.anchor}, which never mentions '
            '${model.hubId} -- the panel would send a reader to another run',
      );
    }
  });

  test('a card names a checkpoint the repository knows, owner included', () {
    // The section names the repository but not always its owner --
    // MEASUREMENTS.md writes `functiongemma-270m-it` bare -- so the test
    // above would accept `another-owner/Qwen3-0.6B`, a re-upload nobody ran,
    // and the panel's Hub link would go to it. README names all of them in
    // full.
    for (final model in WhyItExists.measured) {
      expect(
        readme,
        contains(model.hubId),
        reason:
            '${model.name} links to huggingface.co/${model.hubId}, which this '
            'repository never names -- the card would point at a checkpoint '
            'no run here used',
      );
    }
  });

  test('every card names a Hub repository as owner/name', () {
    for (final model in WhyItExists.measured) {
      expect(
        model.hubId,
        matches(RegExp(r'^[\w.-]+/[\w.-]+$')),
        reason: '${model.name} would build a broken huggingface.co URL',
      );
    }
  });

  test('a revision belongs to the section its own card links to', () {
    // Not "appears somewhere in the file": every revision in MEASUREMENTS.md
    // would satisfy that, so two cards with their revisions swapped would
    // pass while the panel told a reader the wrong commit.
    for (final model in WhyItExists.measured) {
      final revision = model.revision;
      if (revision == null) continue;
      expect(
        revision,
        matches(RegExp(r'^[0-9a-f]{8}$')),
        reason: '${model.name} carries a revision that is not a short sha',
      );
      expect(
        sections[model.anchor],
        contains(revision),
        reason:
            '${model.name} pins $revision, which the section it links to '
            '(#${model.anchor}) does not mention -- the card would name a '
            'commit that run did not record',
      );
    }
  });
}
