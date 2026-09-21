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

/// Every anchor a link could land on: `##` and `###` alike.
Set<String> _anchors(List<String> lines) => {
  for (final line in lines)
    if (RegExp(r'^#{2,3} (.+)$').firstMatch(line) case final heading?)
      anchorFor(heading.group(1)!.trim()),
};

/// What a reader sees after following a link to a `##` heading: that section
/// and the `###` subsections under it, which is where most of the detail
/// lives -- the FunctionGemma run names its checkpoint in a subsection of
/// "The headline numbers", not under the heading itself.
Map<String, String> _sectionsByAnchor(List<String> lines) {
  final sections = <String, StringBuffer>{};
  StringBuffer? current;
  for (final line in lines) {
    if (RegExp(r'^## (.+)$').firstMatch(line) case final heading?) {
      current = sections[anchorFor(heading.group(1)!.trim())] = StringBuffer();
    }
    current?.writeln(line);
  }
  return {for (final e in sections.entries) e.key: e.value.toString()};
}

void main() {
  final lines = _measurements().readAsLinesSync();
  final anchors = _anchors(lines);
  final sections = _sectionsByAnchor(lines);

  test('MEASUREMENTS.md has the sections the cards link to', () {
    expect(anchors, isNotEmpty, reason: 'no headings were found at all');
    for (final model in WhyItExists.measured) {
      expect(
        anchors,
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
