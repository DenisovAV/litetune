import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import 'llm_models.dart';
import 'measured_cards.dart';

/// `/models/embeddings`: EmbeddingGemma 2, the one embedding model litetune
/// fine-tunes, converts and verifies.
///
/// The same card, panel and note as the language-model half and nothing
/// else. It used to carry three sections of its own under the card -- what
/// each command does, four tables of scores and a list of what was not
/// measured -- which put numbers on `/models` that the other half keeps on
/// `/measurements`, and repeated what README.md and MEASUREMENTS.md already
/// say in full. A card carries no score on either half; the link under it
/// goes to the section with the tables.
///
/// The card's facts come from MEASUREMENTS.md: the checkpoint and revision
/// from *The graft*, the held-out count from *litetune end to end*, and the
/// three runtimes from the sections that name EmbeddingGemma 2 and that
/// runtime. `models_page_test.dart` checks every numeral on this half against
/// the EmbeddingGemma 2 sections of MEASUREMENTS.md and README.md.
class EmbeddingModels extends StatelessComponent {
  const EmbeddingModels({super.key});

  /// EmbeddingGemma 2 as a card, in the shape the language models use.
  ///
  /// The revision is the checkpoint's, as MEASUREMENTS.md pins it in *The
  /// graft* (`google/embeddinggemma-2` at `914f7f89`). The anchor is the
  /// section that introduces the path and holds the end-to-end runs.
  ///
  /// The CPU link is that section too, not its *litetune end to end*
  /// subsection: the subsection names neither the checkpoint nor the CPU,
  /// while the section names both -- LiteRT-LM 0.18.0's `EmbeddingEngine` on
  /// the CPU backend of an x86-64 Linux machine.
  static const MeasuredRun embeddingGemma2 = (
    name: 'EmbeddingGemma 2',
    how: 'centroid decider, 2,617 held-out rows',
    what:
        "270M parameters: the text tower of Google's EmbeddingGemma 2. An app "
        "uses it through flutter_edge_ai's EmbeddingModel.createDecider, which "
        'embeds the input and answers with the option whose examples have the '
        'nearest centroid. litetune fine-tunes the tower on one decision and '
        "writes the result into Google's own .litertlm file, so the app loads "
        'one file and nothing ships beside it.',
    hubId: 'google/embeddinggemma-2',
    revision: '914f7f89',
    anchor: _section,
    runtimes: [
      (target: RunTarget.cpu, anchor: _section),
      (target: RunTarget.macGpu, anchor: 'the-cpu-and-the-gpu-on-a-mac'),
      (
        target: RunTarget.qualcommNpu,
        anchor: 'embeddinggemma-2-on-a-qualcomm-npu',
      ),
    ],
  );

  static const _section =
      'embeddinggemma-2-a-decider-fine-tuned-and-grafted-into-googles-file';

  /// The cards, in order. One today.
  static const measured = <MeasuredRun>[embeddingGemma2];

  /// What the runs behind the card covered, as the language-model note does
  /// for its twelve: the sides and the runtimes, no score.
  static const summary =
      "EmbeddingGemma 2's int8 and int4 files were compared with the float32 "
      "fine-tune on held-out rows through LiteRT-LM's EmbeddingEngine on an "
      "x86 host's CPU; the int8 file also on a Mac's CPU and GPU and, "
      "compiled for SM8750, on a phone's NPU. ";

  @override
  Component build(BuildContext context) {
    return MeasuredCards(
      page: '/models/embeddings',
      label: 'Embedding models measured end to end so far',
      runs: measured,
      note: [
        Component.text(summary),
        a(href: '/measurements#${embeddingGemma2.anchor}', [
          Component.text(LlmModels.numbersLink),
        ]),
      ],
    );
  }
}
