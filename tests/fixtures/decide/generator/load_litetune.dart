// Loads an artifact litetune wrote with flutter_edge_ai's own loadDecider (the
// copied lib/) on an embedder whose installed file is the one the artifact
// names, and answers each probe with choose.
//
// Run: dart run bin/load_litetune.dart <artifact.json> <probes.json> <out.json>

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_edge_ai/src/decision/decider.dart';
import 'package:flutter_edge_ai/src/domain/platform_types.dart';
import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_host.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_input.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_modality.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_pass.dart';
import 'package:flutter_edge_ai/src/embedding/task_type.dart';

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
    if (taskType != TaskType.classification) throw StateError('$taskType');
    return List.of(table[input.text]!);
  }

  @override
  Future<String> get unexpectedExit => Completer<String>().future;

  @override
  Future<void> close() async {}
}

Future<void> main(List<String> args) async {
  final artifact =
      jsonDecode(File(args[0]).readAsStringSync()) as Map<String, Object?>;
  final probes = (jsonDecode(File(args[1]).readAsStringSync()) as Map)
      .map((k, v) => MapEntry(k as String, [for (final x in v as List) (x as num).toDouble()]));
  final model = CommonEmbeddingModel.withHost(
    TableHost(artifact['dimension']! as int, probes),
    activeBackend: PreferredBackend.cpu,
    installedFiles: [
      (name: 'embeddinggemma-2-text-270m.litertlm', bytes: 164626432),
    ],
  );
  final decider = await model.loadDecider(artifact);
  final n = (artifact['options']! as List).length;
  final answers = [];
  for (final text in probes.keys) {
    final a = await decider.choose(DeciderInput(EmbeddingInput.text(text)), topK: n);
    answers.add({
      'text': text,
      'choice': a.choice,
      'similarity': a.provenance.similarity,
      'scores': {for (final r in a.ranked) r.id: r.score},
      'probabilities': a.probabilities,
    });
  }
  File(args[2]).writeAsStringSync(jsonEncode(answers));
  stdout.writeln('loaded ${decider.options.length} options, answered ${answers.length}');
}
