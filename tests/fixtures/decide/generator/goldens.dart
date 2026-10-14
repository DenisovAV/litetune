// Writes the goldens litetune's `decide` is held to: the decider code of
// DenisovAV/flutter_gemma (branch feat/decision-engine, at the commit the
// fixtures' README.md names), copied verbatim into lib/, run on fixed vectors.
//
// Every case goes through the public path an app takes -- a
// CommonEmbeddingModel built with installed files and the CPU backend,
// createDecider, toArtifact, choose -- and, beside it, through the fit's own
// functions on the same unit rows, so the intermediate float64 values are
// recorded too.
//
// Run, from the package README.md describes: dart run bin/goldens.dart <out dir>

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_edge_ai/src/decision/decider.dart';
import 'package:flutter_edge_ai/src/decision/decider_embedding.dart'
    show unitRowInto;
import 'package:flutter_edge_ai/src/decision/decider_fit.dart';
import 'package:flutter_edge_ai/src/decision/decider_method.dart';
import 'package:flutter_edge_ai/src/decision/decider_option.dart';
import 'package:flutter_edge_ai/src/decision/embedding_decider.dart';
import 'package:flutter_edge_ai/src/decision/softmax_regression.dart';
import 'package:flutter_edge_ai/src/domain/platform_types.dart';
import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_host.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_input.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_modality.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_pass.dart';
import 'package:flutter_edge_ai/src/embedding/task_type.dart';

const installedName = 'embeddinggemma-2-text-270m.litertlm';
const installedBytes = 164626432;

/// Serves a fixed table, as an engine host would: raw vectors, no prefix.
final class TableHost implements EmbeddingHost {
  TableHost(this.dimension, this.table);
  final int dimension;
  final Map<String, List<double>> table;

  @override
  EmbeddingPassInfo get info => EmbeddingPassInfo(dimension: dimension);

  @override
  Set<EmbeddingModality> get modalities => const {EmbeddingModality.text};

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) async {
    if (taskType != TaskType.classification) {
      throw StateError('embedded with $taskType');
    }
    final v = table[input.text];
    if (v == null) throw StateError('no vector for "${input.text}"');
    return List.of(v);
  }

  @override
  Future<String> get unexpectedExit => Completer<String>().future;

  @override
  Future<void> close() async {}
}

/// fit_fingerprint.dart's 32-bit xorshift (test/decision/fit_fingerprint.dart).
int _next(int x) {
  x ^= (x << 13) & 0xFFFFFFFF;
  x ^= x >> 17;
  x ^= (x << 5) & 0xFFFFFFFF;
  return x & 0xFFFFFFFF;
}

final class Rng {
  Rng(this.state);
  int state;

  /// Uniform in [0, 1).
  double next() {
    state = _next(state);
    return state / 4294967296.0;
  }
}

/// A raw vector as an engine returns one: float32 values.
List<double> f32(List<double> v) => Float32List.fromList(v).toList();

List<double> around(List<double> base, double noise, Rng rng) =>
    f32([for (final b in base) b + noise * (rng.next() - 0.5)]);

/// One case: options, the table, a method and a calibration.
final class Case {
  Case({
    required this.name,
    required this.group,
    required this.dimension,
    required this.options,
    required this.table,
    required this.method,
    required this.calibration,
    this.labelled = const [],
    this.probes = const [],
    this.calibrationJson = const {},
  });

  final String name;

  /// Cases of one group share one vector table, written once.
  final String group;
  final int dimension;
  final List<DeciderOption> options;
  final Map<String, List<double>> table;
  final DeciderMethod method;
  final DeciderCalibration calibration;
  final List<({String text, String optionId})> labelled;
  final List<String> probes;
  final Map<String, Object?> calibrationJson;
}

List<double> listOf(Float64List v) => v.toList();

String b64(Float32List v) {
  final bytes = ByteData(v.length * 4);
  for (var i = 0; i < v.length; i++) {
    bytes.setFloat32(i * 4, v[i], Endian.little);
  }
  return base64.encode(bytes.buffer.asUint8List());
}

Future<Map<String, Object?>> runCase(Case c) async {
  final d = c.dimension;
  final model = CommonEmbeddingModel.withHost(
    TableHost(d, c.table),
    activeBackend: PreferredBackend.cpu,
    installedFiles: [(name: installedName, bytes: installedBytes)],
  );
  final out = <String, Object?>{
    'case': c.name,
    'dimension': d,
    'installedFiles': [
      {'name': installedName, 'bytes': installedBytes},
    ],
    'backend': 'cpu',
    'options': [
      for (final o in c.options)
        {
          'id': o.id,
          'description': ?o.description,
          'examples': [for (final e in o.examples) e.text],
          'metadata': ?o.metadata,
        },
    ],
    'vectors': c.group,
    'method': c.method.name,
    'calibration': c.calibrationJson,
    'labelled': [
      for (final r in c.labelled) {'text': r.text, 'optionId': r.optionId},
    ],
  };

  // The fit's own functions on the unit rows createDecider builds
  // (embedding_decider.dart L163-L179, decider_embedding.dart unitRowInto).
  final texts = <String>[];
  final labels = <int>[];
  final ranks = <int>[];
  for (var o = 0; o < c.options.length; o++) {
    var rank = 0;
    final option = c.options[o];
    if (option.description case final description?) {
      texts.add(description);
      labels.add(o);
      ranks.add(rank++);
    }
    for (final e in option.examples) {
      texts.add(e.text!);
      labels.add(o);
      ranks.add(rank++);
    }
  }
  final x = Float64List(texts.length * d);
  for (var i = 0; i < texts.length; i++) {
    unitRowInto(c.table[texts[i]]!, d, x, i * d, texts[i]);
  }
  final internals = <String, Object?>{
    'rowTexts': texts,
    'labels': labels,
    'ranks': ranks,
  };
  try {
    final c64 = await optionCentroids(
      x,
      Int32List.fromList(labels),
      d,
      c.options.length,
    );
    internals['centroids64'] = listOf(c64);
    if (c.calibration.kind == 'crossFit') {
      final signal = await crossFitSignal(
        x,
        Int32List.fromList(labels),
        Int32List.fromList(ranks),
        d,
        c.options.length,
        null,
      );
      internals['crossFitSignal'] = listOf(signal);
    }
    if (identical(c.method, DeciderMethod.linear)) {
      final fit = await fitSoftmaxRegression(
        x: x,
        labels: Int32List.fromList(labels),
        dimension: d,
        classes: c.options.length,
        decay: linearWeightDecay,
      );
      internals['linear'] = {
        'weights64': listOf(fit.weights),
        'bias64': listOf(fit.bias),
        'loss': fit.loss,
        'iterations': fit.iterations,
        'evaluations': fit.evaluations,
      };
    }
  } catch (e) {
    internals['error'] = '${e.runtimeType}: $e';
  }
  out['internals'] = internals;

  try {
    final decider = await model.createDecider(
      options: c.options,
      method: c.method,
      calibration: c.calibration,
    ) as EmbeddingDecider;
    final artifact = decider.toArtifact();
    // Through JSON, as an app stores it.
    out['artifact'] = jsonDecode(jsonEncode(artifact));
    out['fit'] = {
      'threshold': decider.threshold,
      'temperature': decider.temperature,
      'linearIterations': decider.linearIterations,
      'centroids': b64(decider.centroids),
      'weights': decider.weights == null ? null : b64(decider.weights!),
      'bias': decider.bias?.toList(),
    };
    final loaded = await model.loadDecider(
      jsonDecode(jsonEncode(artifact)) as Map<String, Object?>,
    );
    final answers = <Map<String, Object?>>[];
    for (final p in c.probes) {
      final a = await decider.choose(
        DeciderInput(EmbeddingInput.text(p)),
        topK: c.options.length,
      );
      final b = await loaded.choose(
        DeciderInput(EmbeddingInput.text(p)),
        topK: c.options.length,
      );
      if (a.choice != b.choice ||
          a.provenance.similarity != b.provenance.similarity) {
        throw StateError('the loaded decider answers $p differently');
      }
      answers.add({
        'text': p,
        'choice': a.choice,
        'abstainReason': a.abstainReason?.name,
        'similarity': a.provenance.similarity,
        'margin': a.margin,
        'scoreKind': a.scoreKind.name,
        'confidence': a.confidence,
        'ranked': [
          for (final r in a.ranked)
            {'id': r.id, 'score': r.score, 'probability': r.probability},
        ],
      });
    }
    out['probes'] = answers;
  } catch (e) {
    out['refused'] = {'type': '${e.runtimeType}', 'message': '$e'};
  }
  return out;
}

// -- the cases ------------------------------------------------------------

const d16 = 16;

List<List<double>> bases(Rng rng, int count, int d) => [
  for (var o = 0; o < count; o++)
    [
      for (var j = 0; j < d; j++)
        (j % count == o ? 1.0 : 0.0) + 0.6 * (rng.next() - 0.5),
    ],
];

/// Four options over 16 dimensions: two with a description (seven inputs,
/// two uneven folds), two without (six), metadata on one.
({
  List<DeciderOption> options,
  Map<String, List<double>> table,
  List<({String text, String optionId})> labelled,
  List<String> probes,
})
smallSet(int seed, {double labelledNoise = 1.4}) {
  final rng = Rng(seed);
  final ids = ['alpha', 'beta', 'gamma', 'delta'];
  final base = bases(rng, 4, d16);
  final table = <String, List<double>>{};
  final options = <DeciderOption>[];
  for (var o = 0; o < 4; o++) {
    final described = o.isEven;
    final description = described ? '${ids[o]}: what it means' : null;
    if (description != null) table[description] = around(base[o], 0.5, rng);
    final examples = <EmbeddingInput>[];
    for (var k = 0; k < 6; k++) {
      final text = '${ids[o]} example $k';
      table[text] = around(base[o], 0.9, rng);
      examples.add(EmbeddingInput.text(text));
    }
    options.add(
      DeciderOption(
        ids[o],
        description: description,
        examples: examples,
        metadata: o == 0
            ? {
                'tool': 'open_alpha',
                'args': [1, 2.5, 'x'],
                'nested': {'k': null, 'b': true},
              }
            : null,
      ),
    );
  }
  final labelled = <({String text, String optionId})>[];
  for (var o = 0; o < 4; o++) {
    for (var k = 0; k < 5; k++) {
      final text = 'held ${ids[o]} $k';
      table[text] = around(base[o], labelledNoise, rng);
      // The last row of each option is labelled as the next option: rows the
      // decider answers wrong, without which the temperature has nothing to
      // calibrate on.
      labelled.add((text: text, optionId: ids[k == 4 ? (o + 1) % 4 : o]));
    }
  }
  final probes = <String>[];
  for (var o = 0; o < 4; o++) {
    for (var k = 0; k < 2; k++) {
      final text = 'probe ${ids[o]} $k';
      table[text] = around(base[o], 1.0, rng);
      probes.add(text);
    }
  }
  for (var k = 0; k < 4; k++) {
    final text = 'probe elsewhere $k';
    table[text] = f32([for (var j = 0; j < d16; j++) rng.next() - 0.5]);
    probes.add(text);
  }
  return (options: options, table: table, labelled: labelled, probes: probes);
}

Future<void> main(List<String> args) async {
  final outDir = Directory(args.single)..createSync(recursive: true);
  final cases = <Case>[];
  const seed = 20261014;
  final s = smallSet(seed);

  for (final (method, calibration, json, tag) in [
    (
      DeciderMethod.centroid,
      const DeciderCalibration.crossFit(),
      <String, Object?>{'kind': 'crossFit', 'keep': 0.95},
      'centroid-crossfit',
    ),
    (
      DeciderMethod.linear,
      const DeciderCalibration.crossFit(),
      <String, Object?>{'kind': 'crossFit', 'keep': 0.95},
      'linear-crossfit',
    ),
    (
      DeciderMethod.centroid,
      DeciderCalibration.fromLabelled([
        for (final r in s.labelled) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
      ]),
      <String, Object?>{'kind': 'labelled', 'maxError': 0.05},
      'centroid-labelled',
    ),
    (
      DeciderMethod.linear,
      DeciderCalibration.fromLabelled([
        for (final r in s.labelled) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
      ]),
      <String, Object?>{'kind': 'labelled', 'maxError': 0.05},
      'linear-labelled',
    ),
    (
      DeciderMethod.centroid,
      const DeciderCalibration.minSimilarity(0.25),
      <String, Object?>{'kind': 'minSimilarity', 'threshold': 0.25},
      'centroid-minsimilarity',
    ),
    (
      DeciderMethod.linear,
      const DeciderCalibration.none(),
      <String, Object?>{'kind': 'none'},
      'linear-none',
    ),
    (
      DeciderMethod.linear,
      const DeciderCalibration.crossFit(keep: 0.8),
      <String, Object?>{'kind': 'crossFit', 'keep': 0.8},
      'linear-crossfit-keep08',
    ),
  ]) {
    cases.add(
      Case(
        name: 'small-$tag',
        group: 'small',
        dimension: d16,
        options: s.options,
        table: s.table,
        method: method,
        calibration: calibration,
        calibrationJson: json,
        labelled: json['kind'] == 'labelled' ? s.labelled : const [],
        probes: s.probes,
      ),
    );
  }

  // keep 0.9 over ten held-out inputs: (1 - 0.9) * 10 is 0.9999999999999998,
  // the case keepThreshold adds 1e-9 for (decider_fit.dart L382-L385).
  {
    final rng = Rng(seed + 1);
    final base = bases(rng, 2, d16);
    final table = <String, List<double>>{};
    final options = <DeciderOption>[];
    for (var o = 0; o < 2; o++) {
      final examples = <EmbeddingInput>[];
      for (var k = 0; k < 5; k++) {
        final text = 'k$o-$k';
        table[text] = around(base[o], 1.2, rng);
        examples.add(EmbeddingInput.text(text));
      }
      options.add(DeciderOption('k$o', examples: examples));
    }
    cases.add(
      Case(
        name: 'keep09-ten-inputs',
        group: 'keep09',
        dimension: d16,
        options: options,
        table: table,
        method: DeciderMethod.centroid,
        calibration: const DeciderCalibration.crossFit(keep: 0.9),
        calibrationJson: {'kind': 'crossFit', 'keep': 0.9},
      ),
    );
  }

  // Ties: a group of labelled rows with one vector, one answered right and
  // one wrong, at the lowest signal; a group at the top too. Two examples of
  // one option share a vector.
  {
    final rng = Rng(seed + 2);
    final base = bases(rng, 3, d16);
    final table = <String, List<double>>{};
    final options = <DeciderOption>[];
    final ids = ['t0', 't1', 't2'];
    for (var o = 0; o < 3; o++) {
      final examples = <EmbeddingInput>[];
      for (var k = 0; k < 5; k++) {
        final text = '${ids[o]} ex $k';
        table[text] = k == 4 ? table['${ids[o]} ex 0']! : around(base[o], 0.8, rng);
        examples.add(EmbeddingInput.text(text));
      }
      options.add(DeciderOption(ids[o], examples: examples));
    }
    final labelled = <({String text, String optionId})>[];
    for (var o = 0; o < 3; o++) {
      for (var k = 0; k < 4; k++) {
        final text = 'tie held ${ids[o]} $k';
        table[text] = around(base[o], 1.0, rng);
        labelled.add((text: text, optionId: ids[o]));
      }
    }
    // The low group: the same weak vector, labelled two ways.
    final weak = f32([for (final b in base[0]) 0.3 * b + 1.0 * (rng.next() - 0.5)]);
    table['tie low right'] = weak;
    table['tie low wrong'] = weak;
    labelled.add((text: 'tie low right', optionId: 't0'));
    labelled.add((text: 'tie low wrong', optionId: 't1'));
    // The top group: an exact copy of one centroid direction, labelled two ways.
    table['tie top a'] = f32(base[2]);
    table['tie top b'] = f32(base[2]);
    labelled.add((text: 'tie top a', optionId: 't2'));
    labelled.add((text: 'tie top b', optionId: 't2'));
    for (final (method, tag) in [
      (DeciderMethod.centroid, 'centroid'),
      (DeciderMethod.linear, 'linear'),
    ]) {
      for (final maxError in [0.05, 0.25]) {
        cases.add(
          Case(
            name: 'ties-$tag-maxerror$maxError',
            group: 'ties',
            dimension: d16,
            options: options,
            table: table,
            method: method,
            calibration: DeciderCalibration.fromLabelled([
              for (final r in labelled) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
            ], maxError: maxError),
            calibrationJson: {'kind': 'labelled', 'maxError': maxError},
            labelled: labelled,
          ),
        );
      }
    }
    cases.add(
      Case(
        name: 'ties-crossfit',
        group: 'ties',
        dimension: d16,
        options: options,
        table: table,
        method: DeciderMethod.centroid,
        calibration: const DeciderCalibration.crossFit(),
        calibrationJson: {'kind': 'crossFit', 'keep': 0.95},
      ),
    );
  }

  // Overlap: three options of 7, 6 and 8 inputs -- not multiples of five, so
  // an option's folds are not its rows' positions mod 5 -- with a few training
  // examples drawn near another option, whose best centroid is then that
  // option's fold-centroid; and labelled rows between two options, where the
  // linear layer and the centroids disagree.
  {
    final rng = Rng(seed + 4);
    final base = bases(rng, 3, d16);
    final table = <String, List<double>>{};
    final options = <DeciderOption>[];
    final sizes = [7, 5, 8];
    // Example k of option o drawn near another option.
    const strays = {(0, 2): 1, (0, 5): 2, (2, 3): 0, (1, 4): 0};
    for (var o = 0; o < 3; o++) {
      final description = o == 1 ? 'overlap $o: what it means' : null;
      if (description != null) table[description] = around(base[o], 0.6, rng);
      final examples = <EmbeddingInput>[];
      for (var k = 0; k < sizes[o]; k++) {
        final text = 'overlap $o example $k';
        table[text] = around(base[strays[(o, k)] ?? o], 0.9, rng);
        examples.add(EmbeddingInput.text(text));
      }
      options.add(
        DeciderOption('v$o', description: description, examples: examples),
      );
    }
    final labelled = <({String text, String optionId})>[];
    for (var k = 0; k < 36; k++) {
      final text = 'overlap held $k';
      final a = k % 3, b = (k + 1) % 3;
      final mix = k < 24 ? 0.0 : 0.4;
      table[text] = around(
        [for (var j = 0; j < d16; j++) (1 - mix) * base[a][j] + mix * base[b][j]],
        1.0,
        rng,
      );
      labelled.add((text: text, optionId: 'v$a'));
    }
    for (final (method, tag) in [
      (DeciderMethod.centroid, 'centroid'),
      (DeciderMethod.linear, 'linear'),
    ]) {
      cases.add(
        Case(
          name: 'overlap-$tag-crossfit',
          group: 'overlap',
          dimension: d16,
          options: options,
          table: table,
          method: method,
          calibration: const DeciderCalibration.crossFit(),
          calibrationJson: {'kind': 'crossFit', 'keep': 0.95},
        ),
      );
      cases.add(
        Case(
          name: 'overlap-$tag-labelled',
          group: 'overlap',
          dimension: d16,
          options: options,
          table: table,
          method: method,
          calibration: DeciderCalibration.fromLabelled([
            for (final r in labelled) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
          ], maxError: 0.3),
          calibrationJson: {'kind': 'labelled', 'maxError': 0.3},
          labelled: labelled,
        ),
      );
    }
  }

  // Refusals the fit makes (the argument checks before it are litetune's own).
  {
    // Every labelled row right with room to spare: the temperature's floor.
    final rng = Rng(seed + 3);
    final base = bases(rng, 3, d16);
    final table = <String, List<double>>{};
    final options = <DeciderOption>[];
    for (var o = 0; o < 3; o++) {
      final examples = <EmbeddingInput>[];
      for (var k = 0; k < 3; k++) {
        final text = 'r$o-$k';
        table[text] = around(base[o], 0.2, rng);
        examples.add(EmbeddingInput.text(text));
      }
      options.add(DeciderOption('r$o', examples: examples));
    }
    final clean = <({String text, String optionId})>[];
    for (var o = 0; o < 3; o++) {
      for (var k = 0; k < 3; k++) {
        final text = 'clean $o $k';
        table[text] = around(base[o], 0.05, rng);
        clean.add((text: text, optionId: 'r$o'));
      }
    }
    cases.add(
      Case(
        name: 'refused-temperature-floor',
        group: 'refusals',
        dimension: d16,
        options: options,
        table: table,
        method: DeciderMethod.centroid,
        calibration: DeciderCalibration.fromLabelled([
          for (final r in clean) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
        ]),
        calibrationJson: {'kind': 'labelled', 'maxError': 0.05},
        labelled: clean,
      ),
    );

    // The row with the highest cosine answered wrong, and no error allowed.
    final wrongTop = <({String text, String optionId})>[
      ...clean,
      (text: 'wrong top', optionId: 'r0'),
    ];
    table['wrong top'] = f32(base[1]);
    cases.add(
      Case(
        name: 'refused-no-threshold',
        group: 'refusals',
        dimension: d16,
        options: options,
        table: table,
        method: DeciderMethod.centroid,
        calibration: DeciderCalibration.fromLabelled([
          for (final r in wrongTop) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
        ], maxError: 0.0),
        calibrationJson: {'kind': 'labelled', 'maxError': 0.0},
        labelled: wrongTop,
      ),
    );

    // One vector under every label: a flat distribution fits best.
    final same = f32([for (var j = 0; j < d16; j++) rng.next() - 0.5]);
    final noise = <({String text, String optionId})>[];
    for (var k = 0; k < 30; k++) {
      final text = 'noise $k';
      table[text] = same;
      noise.add((text: text, optionId: 'r${k % 3}'));
    }
    cases.add(
      Case(
        name: 'refused-temperature-ceiling',
        group: 'refusals',
        dimension: d16,
        options: options,
        table: table,
        method: DeciderMethod.centroid,
        calibration: DeciderCalibration.fromLabelled([
          for (final r in noise) (input: EmbeddingInput.text(r.text), optionId: r.optionId),
        ], maxError: 0.95),
        calibrationJson: {'kind': 'labelled', 'maxError': 0.95},
        labelled: noise,
      ),
    );

    // Two inputs that cancel out.
    final v = f32([for (var j = 0; j < d16; j++) rng.next() - 0.5]);
    final cancelTable = <String, List<double>>{
      ...table,
      'up': v,
      'down': [for (final x in v) -x],
    };
    cases.add(
      Case(
        name: 'refused-cancelling-inputs',
        group: 'refusals',
        dimension: d16,
        options: [
          options[0],
          DeciderOption('cancel', examples: [EmbeddingInput.text('up'), EmbeddingInput.text('down')]),
        ],
        table: cancelTable,
        method: DeciderMethod.centroid,
        calibration: const DeciderCalibration.none(),
        calibrationJson: {'kind': 'none'},
      ),
    );
  }

  final tables = <String, Map<String, List<double>>>{};
  for (final c in cases) {
    (tables[c.group] ??= {}).addAll(c.table);
  }
  for (final MapEntry(key: group, value: table) in tables.entries) {
    File('${outDir.path}/vectors-$group.json').writeAsStringSync(jsonEncode(table));
  }
  for (final c in cases) {
    final result = await runCase(c);
    File('${outDir.path}/${c.name}.json').writeAsStringSync(
      jsonEncode(result),
    );
    final fit = result['fit'] as Map<String, Object?>?;
    stdout.writeln(
      '${c.name}: ${result['refused'] ?? 'threshold ${fit?['threshold']}, '
          'temperature ${fit?['temperature']}, iterations ${fit?['linearIterations']}'}',
    );
  }

  await fingerprints(outDir);
  await lbfgs(outDir);

  final ids = {
    'installed': deciderSpaceId(
      CommonEmbeddingModel.withHost(
        TableHost(768, const {}),
        activeBackend: PreferredBackend.cpu,
        installedFiles: [(name: installedName, bytes: installedBytes)],
      ),
    ),
    'installed-escaped': deciderSpaceId(
      CommonEmbeddingModel.withHost(
        TableHost(768, const {}),
        activeBackend: PreferredBackend.cpu,
        installedFiles: [
          (name: 'a;b+c@d%e\u0001\u001f\u007f é 😀 {x}.litertlm', bytes: 12345),
        ],
      ),
    ),
    'embedder-id': deciderSpaceId(
      CommonEmbeddingModel.withHost(
        TableHost(768, const {}),
        activeBackend: PreferredBackend.cpu,
      ),
      embedderId: 'eg2@v1;cpu+int4%\n\t 50% ü',
    ),
  };
  File('${outDir.path}/space-ids.json').writeAsStringSync(
    const JsonEncoder.withIndent(' ').convert({
      'cases': [
        {
          'installedFiles': [
            {'name': installedName, 'bytes': installedBytes},
          ],
          'dimension': 768,
          'backend': 'cpu',
          'spaceId': ids['installed'],
        },
        {
          'installedFiles': [
            {'name': 'a;b+c@d%e\u0001\u001f\u007f é 😀 {x}.litertlm', 'bytes': 12345},
          ],
          'dimension': 768,
          'backend': 'cpu',
          'spaceId': ids['installed-escaped'],
        },
        {
          'embedderId': 'eg2@v1;cpu+int4%\n\t 50% ü',
          'dimension': 768,
          'backend': 'cpu',
          'spaceId': ids['embedder-id'],
        },
      ],
    }),
  );
  stdout.writeln(ids);
}

// -- fit_fingerprint.dart's fixture ------------------------------------------

/// test/decision/fit_fingerprint.dart's `fingerprintRequest`, verbatim but
/// for the function name.
DeciderFitRequest fingerprintRequest(int c, {required bool linear}) {
  const d = 768;
  const k = 10;
  final n = c * k;
  final x = Float64List(n * d);
  final labels = Int32List(n);
  final ranks = Int32List(n);
  var state = 0x9E3779B9 ^ c;
  for (var i = 0; i < n; i++) {
    labels[i] = i ~/ k;
    ranks[i] = i % k;
    var norm = 0.0;
    for (var j = 0; j < d; j++) {
      state = _next(state);
      final v = (j % c == labels[i] ? 1.0 : 0.0) + state / 4294967296.0 - 0.5;
      x[i * d + j] = v;
      norm += v * v;
    }
    norm = math.sqrt(norm);
    for (var j = 0; j < d; j++) {
      x[i * d + j] /= norm;
    }
  }
  return DeciderFitRequest(
    dimension: d,
    optionCount: c,
    optionIds: [for (var o = 0; o < c; o++) 'o$o'],
    inputs: x,
    labels: labels,
    ranks: ranks,
    linear: linear,
    calibration: 'crossFit',
    keep: 0.95,
  );
}

double _weighted(List<double> values) {
  var sum = 0.0;
  for (var i = 0; i < values.length; i++) {
    sum += values[i] * (i % 13 + 1);
  }
  return sum;
}

Future<void> fingerprints(Directory outDir) async {
  final rows = <Map<String, Object?>>[];
  for (final c in [10, 50, 150]) {
    for (final linear in [false, true]) {
      final request = fingerprintRequest(c, linear: linear);
      final fit = await fitDecider(request);
      final row = <String, Object?>{
        'options': c,
        'linear': linear,
        'inputSum': _weighted(request.inputs),
        'threshold': fit.threshold,
        'centroids': _weighted(fit.centroids),
        'weights': fit.weights == null ? null : _weighted(fit.weights!),
        'bias': fit.bias == null ? null : _weighted(fit.bias!),
        'loss': fit.linearLoss,
        'iterations': fit.linearIterations,
      };
      if (c == 10) {
        row['centroidsB64'] = b64(fit.centroids);
        if (linear) {
          row['weightsB64'] = b64(fit.weights!);
          row['biasValues'] = fit.bias!.toList();
          final soft = await fitSoftmaxRegression(
            x: request.inputs,
            labels: request.labels,
            dimension: 768,
            classes: c,
          );
          row['evaluations'] = soft.evaluations;
        }
        row['crossFitSignal'] = (await crossFitSignal(
          request.inputs,
          request.labels,
          request.ranks,
          768,
          c,
          null,
        )).toList();
      }
      rows.add(row);
      stdout.writeln(
        'fingerprint $c ${linear ? 'linear' : 'centroid'}: threshold '
        '${row['threshold']} centroids ${row['centroids']} weights '
        '${row['weights']} bias ${row['bias']} loss ${row['loss']} '
        'iterations ${row['iterations']}',
      );
    }
  }
  File('${outDir.path}/fingerprints.json').writeAsStringSync(
    jsonEncode({'fits': rows}),
  );
}

// -- minimizeLbfgs on its own --------------------------------------------------

/// The chained Rosenbrock function and its gradient, written so that every
/// operation is one a Python float does in the same order.
double rosenbrock(Float64List x, Float64List g) {
  g.fillRange(0, g.length, 0);
  var f = 0.0;
  for (var i = 0; i < x.length - 1; i++) {
    final a = x[i + 1] - x[i] * x[i];
    final b = 1 - x[i];
    f = f + (100 * a * a + b * b);
    g[i] = g[i] + (-400 * a * x[i] - 2 * b);
    g[i + 1] = g[i + 1] + 200 * a;
  }
  return f;
}

/// `(x_i - (i + 1))^4 + 0.001 x_i^2`, summed: flat near its minimum.
double quartic(Float64List x, Float64List g) {
  var v = 0.0;
  for (var i = 0; i < x.length; i++) {
    final xi = x[i];
    final c = xi - (i + 1);
    v = v + (c * c * c * c + 0.001 * xi * xi);
    g[i] = 4 * c * c * c + 0.002 * xi;
  }
  return v;
}

/// `x_i^4 - 3 x_i^2 + x_i`, summed: two wells in every coordinate.
double well(Float64List x, Float64List g) {
  var v = 0.0;
  for (var i = 0; i < x.length; i++) {
    final xi = x[i];
    v = v + (xi * xi * xi * xi - 3 * xi * xi + xi);
    g[i] = 4 * xi * xi * xi - 6 * xi + 1;
  }
  return v;
}

/// `1e12 (x_0 - 1)^2 + sum_{i>0} (x_i - 2)^4`: one coordinate far stiffer.
double mix(Float64List x, Float64List g) {
  var v = 0.0;
  for (var i = 0; i < x.length; i++) {
    final xi = x[i];
    if (i == 0) {
      final c = xi - 1;
      v = v + 1e12 * c * c;
      g[i] = 2e12 * c;
    } else {
      final c = xi - 2;
      v = v + c * c * c * c;
      g[i] = 4 * c * c * c;
    }
  }
  return v;
}

const objectives = <String, double Function(Float64List, Float64List)>{
  'rosenbrock': rosenbrock,
  'quartic': quartic,
  'well': well,
  'mix': mix,
};

Future<void> lbfgs(Directory outDir) async {
  final runs = <Map<String, Object?>>[];
  for (final (name, objective, start, settings) in [
    ('rosenbrock-2', 'rosenbrock', [-1.2, 1.0], <String, num>{}),
    ('rosenbrock-10', 'rosenbrock', List<double>.filled(10, -1.0), <String, num>{}),
    ('max-iter-5', 'rosenbrock', [-1.2, 1.0], <String, num>{'maxIter': 5}),
    ('max-eval-12', 'rosenbrock', [-1.2, 1.0], <String, num>{'maxEval': 12}),
    ('history-3', 'rosenbrock', List<double>.filled(10, -1.0), <String, num>{'historySize': 3}),
    ('tolerance-change-1e-3', 'rosenbrock', [-1.2, 1.0], <String, num>{'toleranceChange': 1e-3}),
    ('tolerance-grad-1e-2', 'rosenbrock', [-1.2, 1.0], <String, num>{'toleranceGrad': 1e-2}),
    ('lr-0.5', 'rosenbrock', [-1.2, 1.0], <String, num>{'lr': 0.5}),
    ('far-start', 'rosenbrock', [3.0, -4.0, 2.5], <String, num>{}),
    ('at-the-minimum', 'rosenbrock', [1.0, 1.0], <String, num>{}),
    // Found to reach a branch the runs above do not (litetune's port, each
    // branch changed in turn): a curvature pair below 1e-10, a line search
    // capped by the evaluations left, the sufficient-decrease constant, the
    // step-size stop, the zoom's insufficient-progress rule.
    ('quartic-tiny-curvature', 'quartic', [-1.25, -1.75], <String, num>{'toleranceGrad': 1e-30, 'toleranceChange': 1e-30}),
    ('rosenbrock-max-eval-20', 'rosenbrock', [1.75, -2.0, 4.0], <String, num>{'maxEval': 20}),
    ('quartic-history-2', 'quartic', [2.25, -1.5], <String, num>{'historySize': 2}),
    ('mix-step-size', 'mix', [-9.0, -4.0], <String, num>{}),
    ('well-zoom', 'well', [-2.5, -2.0, 2.75, -1.0], <String, num>{'toleranceGrad': 1e-30, 'toleranceChange': 1e-30}),
  ]) {
    final x = Float64List.fromList(start);
    final f = objectives[objective]!;
    final result = await minimizeLbfgs(
      (at, g) async => f(at, g),
      x,
      lr: (settings['lr'] ?? 1).toDouble(),
      maxIter: (settings['maxIter'] ?? 500).toInt(),
      maxEval: settings['maxEval']?.toInt(),
      toleranceGrad: (settings['toleranceGrad'] ?? 1e-7).toDouble(),
      toleranceChange: (settings['toleranceChange'] ?? 1e-9).toDouble(),
      historySize: (settings['historySize'] ?? 20).toInt(),
    );
    runs.add({
      'name': name,
      'objective': objective,
      'start': start,
      'settings': settings,
      'x': x.toList(),
      'loss': result.loss,
      'iterations': result.iterations,
      'evaluations': result.evaluations,
    });
    stdout.writeln('lbfgs $name: loss ${result.loss} iterations '
        '${result.iterations} evaluations ${result.evaluations}');
  }
  File('${outDir.path}/lbfgs.json').writeAsStringSync(
    const JsonEncoder.withIndent(' ').convert({'runs': runs}),
  );
}
