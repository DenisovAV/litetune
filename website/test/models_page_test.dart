// The embeddings half of `/models` carries numbers, which the language-model
// cards do not. Each has to be one a reader can find in the files the page
// says it comes from, so this checks every numeral on it against the
// EmbeddingGemma 2 sections of MEASUREMENTS.md and README.md -- scoped to
// those sections, because "somewhere in a 2,500-line file" is a bar any
// plausible number clears.
import 'dart:io';

import 'package:litetune_website/measurements/measurements_page.dart';
import 'package:litetune_website/models/embedding_models.dart';
import 'package:litetune_website/models/llm_models.dart';
import 'package:litetune_website/models/measured_cards.dart';
import 'package:litetune_website/models/models_page.dart';
import 'package:test/test.dart';

Directory _root() {
  for (final candidate in [Directory.current.parent, Directory.current]) {
    if (File('${candidate.path}/MEASUREMENTS.md').existsSync()) {
      return candidate;
    }
  }
  throw StateError(
    'cannot find MEASUREMENTS.md from ${Directory.current.path}',
  );
}

/// The text from the line [from] starts to the line [to] starts, exclusive.
///
/// A section that is the last in its file runs to the end of it, so a missing
/// [to] is that; a missing [from] is a renamed section and fails.
String _between(String text, String from, String to) {
  final start = text.indexOf(from);
  if (start < 0) throw StateError('no "$from" in the file');
  var end = text.indexOf(to, start + from.length);
  if (end < 0) end = text.length;
  return text.substring(start, end);
}

/// Every string the embeddings half renders, card and panel included.
List<String> _embeddingsCopy() => [
  for (final run in EmbeddingModels.measured) ...[run.name, run.how, run.what],
  EmbeddingModels.summary,
  for (final step in EmbeddingModels.steps) ...[step.command, step.text],
  for (final table in EmbeddingModels.tables) ...[
    table.title,
    table.conditions,
    ...table.header,
    for (final row in table.rows) ...row,
    ...table.notes,
  ],
  ...EmbeddingModels.notMeasured,
];

String _normalise(String text) =>
    text.replaceAll('−', '-').replaceAll(' ', ' ');

void main() {
  final root = _root();
  final measurements = File('${root.path}/MEASUREMENTS.md').readAsStringSync();
  final readme = File('${root.path}/README.md').readAsStringSync();

  // The three sections the page names: the one that introduces the path and
  // holds the end-to-end runs, the NPU run, and the Mac section, whose last
  // rows are EmbeddingGemma 2's GPU run.
  final measurementsSource = _between(
    measurements,
    "## EmbeddingGemma 2: a decider fine-tuned and grafted into Google's file",
    '\n## ',
  );
  final npu = _between(
    measurements,
    '## EmbeddingGemma 2 on a Qualcomm NPU',
    '\n## ',
  );
  final mac = _between(
    measurements,
    '## The CPU and the GPU on a Mac',
    '\n## ',
  );
  final readmeSource = _between(
    readme,
    '### Decisions on EmbeddingGemma 2',
    '### Gemma 4 towers',
  );
  final source = _normalise(
    [measurementsSource, npu, mac, readmeSource].join('\n'),
  );

  test('every numeral on the embeddings page is in its source sections', () {
    final numeral = RegExp(r'[+\-]?\d[\d,]*(?:\.\d+)?');
    for (final text in _embeddingsCopy()) {
      for (final match in numeral.allMatches(_normalise(text))) {
        var token = match.group(0)!;
        if (token.endsWith(',')) token = token.substring(0, token.length - 1);
        expect(
          source,
          contains(token),
          reason:
              '"$token" in "$text" is in neither the EmbeddingGemma 2 sections '
              'of MEASUREMENTS.md nor those of README.md',
        );
      }
    }
  });

  test('a table carries its conditions and rows as wide as its header', () {
    for (final table in EmbeddingModels.tables) {
      expect(table.conditions.trim(), isNotEmpty, reason: table.title);
      for (final row in table.rows) {
        expect(row, hasLength(table.header.length), reason: table.title);
      }
    }
  });

  test('the embedding card links to an id the measurements page emits', () {
    final ids = MeasurementsPage.idsIn(measurements);
    for (final run in EmbeddingModels.measured) {
      expect(ids, contains(run.anchor));
      expect(readme, contains(run.hubId));
      expect(run.what, matches(RegExp(r'^\d+(\.\d+)?[MB]\b')));
      final revision = run.revision;
      if (revision != null) {
        expect(measurementsSource, contains(revision));
      }
    }
  });

  test('no two cards on /models share a panel id', () {
    final slugs = [
      for (final run in [...LlmModels.measured, ...EmbeddingModels.measured])
        MeasuredCards.slugFor(run),
    ];
    expect(slugs.toSet(), hasLength(slugs.length));
    expect(slugs, isNot(contains(MeasuredCards.closeTarget)));
  });

  test('the switch links the two routes main.server.dart registers', () {
    expect(ModelsTab.llm.path, '/models');
    expect(ModelsTab.embeddings.path, '/models/embeddings');
    final server = File(
      '${root.path}/website/lib/main.server.dart',
    ).readAsStringSync();
    for (final tab in ModelsTab.values) {
      expect(server, contains('ModelsTab.${tab.name}'));
    }
  });
}
