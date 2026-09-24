// The panel's links, checked against the files they point into.
//
// Each card links to a section of the `/measurements` page by its heading id.
// That page is `MEASUREMENTS.md` rendered, so a renamed heading takes its own
// id with it -- but the card still names the section it wants, and a card
// pointing at an id the page does not emit is a link that lands on the top of
// a 1000-line page instead of the run it promised, silently, on the live site.
// So the ids are taken from the page's own spelling rather than from a second
// copy of it here.
//
// The same check covers the Hub links by their shape only: whether
// huggingface.co still serves a repository is not something a unit test can
// answer, and a test that pretends to would be worse than none.
import 'dart:io';

import 'package:litetune_website/landing/sections/why_it_exists.dart';
import 'package:litetune_website/measurements/measurements_page.dart';
import 'package:test/test.dart';

/// The page's own spelling, not a second copy of it. There used to be one
/// here, which meant a test could agree with itself and disagree with the
/// site.
String anchorFor(String heading) => MeasurementsPage.headingId(heading);

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
  _slugs();
  final sections = _sectionsByAnchor(_measurements().readAsLinesSync());
  final readme = _readme().readAsStringSync();

  test('the measurements page emits the ids the cards link to', () {
    // Against the ids the page renders, not the headings a file happens to
    // have: the card's href and this list now come out of one function, so a
    // heading that is renamed fails here rather than on the live site.
    final ids = MeasurementsPage.idsIn(_measurements().readAsStringSync());
    expect(ids, isNotEmpty, reason: 'the page emitted no heading ids at all');
    for (final model in WhyItExists.measured) {
      expect(
        ids,
        contains(model.anchor),
        reason:
            '${model.name} links to /measurements#${model.anchor}, which is '
            'not an id that page emits',
      );
    }
  });

  test('every panel says what its model is and what it is for', () {
    // What a reader opens a card for. It is not on the card face -- that
    // carries the model and how it was scored -- so an empty one renders as a
    // blank paragraph above the checkpoint row rather than as an absence
    // anyone would notice. A record field cannot be omitted in Dart; it can be
    // left empty.
    for (final model in WhyItExists.measured) {
      expect(
        model.what.trim(),
        isNotEmpty,
        reason: '${model.name} carries no description',
      );
      // Size first, because it is what a reader is choosing between. The
      // punctuation after it is not pinned -- this caught the move from a
      // one-line card face to a paragraph in the panel, which is a change of
      // shape and not of rule.
      expect(
        model.what,
        matches(RegExp(r'^(About )?\d+(\.\d+)?[MB]\b')),
        reason: '${model.name} does not open with its size: ${model.what}',
      );
      // It reads in the panel now, not on the card face, and it is meant to
      // say enough to choose on. A one-liner that drifted back here would
      // render as a lonely sentence above the checkpoint row.
      expect(
        model.what.length,
        greaterThan(140),
        reason: '${model.name} says too little to be worth opening',
      );
    }
  });

  test('each card links to a section that measured a conversion', () {
    // Naming the checkpoint is not enough: a model can appear in more than one
    // section, and only some of them convert anything. The Gemma 4 card was
    // filed against a section comparing two float checkpoints -- which says so
    // itself, "no conversion between them" -- while sitting under the label
    // "Measured end to end so far". A cost-of-conversion row is what every
    // section behind these cards has and a training comparison does not.
    for (final model in WhyItExists.measured) {
      expect(
        sections[model.anchor]?.toLowerCase(),
        contains('cost of conversion'),
        reason:
            '${model.name} links to #${model.anchor}, which has no '
            'cost-of-conversion row -- the card promises a conversion the '
            'section it opens did not measure',
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

// The card is a link to its own panel, so the panel's id is what makes the
// two meet. Nothing in a Dart build notices two elements sharing an id: the
// browser opens the first, and the second card silently shows the first
// card's checkpoint. The id is derived from the Hub id's last segment, which
// is unique across today's six and is not guaranteed to stay that way --
// `google/gemma-3-1b-it` and `someone-else/gemma-3-1b-it` would collide.
void _slugs() {
  test('closing a panel lands somewhere, and not on another panel', () {
    // Both close links point at `#${WhyItExists.closeTarget}`, and `:target`
    // stops matching when the fragment names nothing -- so a container id that
    // drifts from the close links leaves a panel that cannot be closed, with
    // every test green. The second half matters too: if the close target were
    // also a panel id, closing one panel would open another.
    expect(WhyItExists.closeTarget, isNotEmpty);
    expect(
      [for (final model in WhyItExists.measured) WhyItExists.slugFor(model)],
      isNot(contains(WhyItExists.closeTarget)),
      reason: 'closing a panel would open another one',
    );
  });

  test('each card opens its own panel and no one else\'s', () {
    final slugs = [
      for (final model in WhyItExists.measured) WhyItExists.slugFor(model),
    ];
    expect(
      slugs.toSet().length,
      slugs.length,
      reason: 'two cards share a panel id: $slugs',
    );
  });

  test('a panel id is usable as a URL fragment', () {
    for (final model in WhyItExists.measured) {
      final slug = WhyItExists.slugFor(model);
      expect(
        slug,
        matches(RegExp(r'^[a-z][a-z0-9._-]*$')),
        reason: '${model.name} produces a fragment a browser cannot target',
      );
    }
  });
}
