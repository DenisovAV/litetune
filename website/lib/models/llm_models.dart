import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import 'measured_cards.dart';

/// The language models measured end to end: the default half of `/models`.
///
/// They used to sit under the paragraph on the landing page that explains why
/// the tool exists; one card per family made that page grow with every
/// release, so they live here and the landing page links to them.
///
/// The cards hold to the rule that paragraph holds to: it carries no numbers,
/// because the measured figures live in MEASUREMENTS.md, where a reader can
/// see the intervals and the sample sizes next to them. The only quantity on them is a
/// sample size, which is the thing a claim needs rather than a claim itself,
/// and no card carries a score or a verdict. That is not squeamishness:
/// `MEASUREMENTS.md` records the FunctionGemma conversion cost resolving in
/// 2 of 6 recipe-runs across three runs of the same configuration, calls
/// whether 640 examples resolve it "close to a coin flip", and says in terms
/// that one run's verdict should not be quoted as the answer. A card is the
/// shortest place there is to quote a verdict, so it quotes none.
///
/// A card carries the model, the scorer and the sample size, and stops there.
/// It used to carry a third line apiece -- a phone run, a refused base, an
/// export that needed no flag -- and four cards' worth of those read as
/// footnotes rather than as one fact each, with nothing shared to compare
/// across. What is worth saying about a single run is worth a sentence in
/// `MEASUREMENTS.md`, which the link below the cards goes to.
///
/// The note under the cards keeps the two execution sides separate: every
/// headline candidate ran on LiteRT-LM's CPU backend, while some float
/// references ran on CUDA. Four-bit results cover four of the fine-tuned
/// models rather than all twelve (Gemma 4's is on its base weights). The separate Qwen3 phone run is named because its CPU and GPU results
/// resolve in opposite directions between the eight-bit and mixed four-bit
/// recipes; a blanket "four bits cost more" would therefore be false.
///
/// The size is in the card name because `models.py` scopes the `gemma-3-text`
/// family to the 270M and the 1B, and both are now measured -- two cards that
/// differ in nothing but size. Qwen3's cards carry their sizes because its rule
/// is scoped to the sizes that were run, and Qwen2.5's for the same reason,
/// where the rule is scoped to the sizes *and* variant.
class LlmModels extends StatelessComponent {
  const LlmModels({super.key});

  static const measurementSummary =
      'Across the twelve headline runs, every converted candidate ran on '
      "LiteRT-LM's CPU backend; some float references ran on CUDA. The "
      'eight-bit gap was small on all twelve and often unresolved. Four of the '
      'fine-tuned models '
      'also have blockwise four-bit results with the candidates on LiteRT-LM\'s '
      'CPU backend, where the gap was larger. A separate Qwen3 phone run shows '
      'that the backend can change the answer. ';

  @override
  Component build(BuildContext context) {
    return MeasuredCards(
      page: '/models',
      label: 'Language models measured end to end so far',
      runs: measured,
      note: [
        Component.text(measurementSummary),
        a(href: '/measurements', [
          Component.text('The numbers, and what they do not establish'),
        ]),
      ],
    );
  }

  /// The runs, in the order the cards show them.
  static const measured = <MeasuredRun>[
    (
      name: 'FunctionGemma 270M',
      how: 'tool-call scoring, 640 held-out rows',
      what:
          '270M parameters, text only. Google built it for tool calls: '
          'given a list of functions it picks one and fills in its '
          'arguments, which is the job an app needs done on the phone '
          'rather than in a round trip to a server. Small enough that '
          'the whole model ships inside the app.',
      hubId: 'google/functiongemma-270m-it',
      revision: null,
      anchor: 'the-headline-numbers',
    ),
    (
      name: 'Gemma 3 270M',
      how: 'exact-text scoring, 600 held-out rows',
      what:
          '270M parameters, text only — the smallest checkpoint '
          'measured here. In the banking77 run it was fine-tuned on 2,400 '
          'examples and evaluated as a 77-way intent classifier, rather than '
          'as a conversational model.',
      hubId: 'google/gemma-3-270m-it',
      revision: 'ac82b4e8',
      anchor: 'a-second-family-and-the-second-scorer',
    ),
    (
      name: 'Gemma 3 1B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '1B parameters, text only. It was measured on the same banking77 '
          'split and through the same export rule as the 270M, providing a '
          'second size in one family without assuming that size predicts the '
          'conversion cost.',
      hubId: 'google/gemma-3-1b-it',
      revision: 'dcc83ea8',
      anchor: 'the-same-family-four-times-the-size',
    ),
    (
      name: 'Qwen3 0.6B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '0.6B parameters, text only. A reasonable first try when the '
          'task is labels, and the first family measured here that '
          'litetune had no per-model rule for — it exported with no flag '
          'from litetune; since 0.4.3 its rule hands the base '
          'model, and a checkpoint litetune trained on rendered prompts, '
          "LiteRT-LM's own Qwen 3 template.",
      hubId: 'Qwen/Qwen3-0.6B',
      revision: 'c1899de2',
      anchor: 'a-fourth-family-and-the-first-that-is-not-gemma',
    ),
    (
      name: 'Qwen3 1.7B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '1.7B parameters, text only, and the same chat template as the '
          "0.6B byte for byte, so its rule hands it LiteRT-LM's own Qwen 3 "
          'template on the same terms. Measured on litert-lm 0.18.0, with '
          'neither eight-bit conversion cost resolved.',
      hubId: 'Qwen/Qwen3-1.7B',
      revision: '70d244cc',
      anchor: 'the-two-qwen-families-one-size-up',
    ),
    (
      name: 'Qwen2.5 0.5B Instruct',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '0.5B parameters, text only. It is the one checkpoint here on '
          'which channelwise four bits produced a score at all, where the '
          'others either refused or came apart, and those scores cost it '
          '+0.2683 and +0.2500 against its float 0.7700 on litert-lm 0.17.1.',
      hubId: 'Qwen/Qwen2.5-0.5B-Instruct',
      revision: '7ae55760',
      anchor:
          'a-fifth-family-and-the-first-where-channelwise-four-bits-answered',
    ),
    (
      name: 'Qwen2.5 1.5B Instruct',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '1.5B parameters, text only, and the same chat template as the '
          '0.5B Instruct byte for byte, composed behind the same text-parts '
          'block. Measured on litert-lm 0.18.0 with eight bits only, '
          'neither conversion cost resolved.',
      hubId: 'Qwen/Qwen2.5-1.5B-Instruct',
      revision: '989aa798',
      anchor: 'the-two-qwen-families-one-size-up',
    ),
    (
      name: 'Qwen3.5 0.8B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '0.8B parameters, text only, and most of its layers are not '
          'attention: 18 of its 24 are linear attention. It takes '
          "LiteRT-LM's own Qwen 3.5 template, edited to render a single "
          'turn as the checkpoint does, and verify gives every prompt an '
          'Engine of its own.',
      hubId: 'Qwen/Qwen3.5-0.8B',
      revision: '2fc06364',
      anchor: 'three-more-families-granite-41-minicpm5-and-qwen-35',
    ),
    (
      name: 'Qwen3.5 2B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '2B parameters, text only, under the same rule as the 0.8B: its '
          'bundle carries the same edited template, byte for byte. Measured '
          'on litert-lm 0.18.0, where its eight-bit dynamic conversion '
          'cost resolved.',
      hubId: 'Qwen/Qwen3.5-2B',
      revision: '15852e8c',
      anchor: 'three-more-families-granite-41-minicpm5-and-qwen-35',
    ),
    (
      name: 'MiniCPM5 1B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          "1B parameters, text only. It takes LiteRT-LM's own MiniCPM5 "
          'template, edited for the start token its checkpoint template '
          'prints itself: the runtime puts that token in front of what it '
          'renders, and the first export of it could not check that.',
      hubId: 'openbmb/MiniCPM5-1B',
      revision: '87179e5c',
      anchor: 'three-more-families-granite-41-minicpm5-and-qwen-35',
    ),
    (
      name: 'Granite 4.1 3B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '3.4B parameters, text only, and the largest text-only model '
          "measured here. Its own chat template, composed behind litetune's "
          'text-parts block. Measured on litert-lm 0.18.0 with eight bits '
          'only, neither conversion cost resolved.',
      hubId: 'ibm-granite/granite-4.1-3b',
      revision: 'c0650403',
      anchor: 'three-more-families-granite-41-minicpm5-and-qwen-35',
    ),
    (
      name: 'Gemma 4 E2B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          'About 5B parameters, and multimodal — a text tower with '
          'vision and audio towers beside it. For an assistant that has '
          'to see or hear as well as read. A LoRA run on it is scoped '
          'to the text tower, because all three use the same projection '
          'names and a run scoped by name alone reaches every one of '
          'them.',
      hubId: 'google/gemma-4-E2B-it',
      // Null although MEASUREMENTS.md names a commit: that run passed no
      // `--revision` and the hash was read back from the cache afterwards.
      // Showing it here in the same slot as ten pinned ones would give it a
      // parity the section it links to explicitly denies.
      revision: null,
      anchor: 'end-to-end-on-the-seven-projection-checkpoint',
    ),
  ];
}
