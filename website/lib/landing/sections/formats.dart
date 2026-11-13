import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// What comes out of the pipeline, and what it is compatible with.
///
/// Three rows carry a qualifier in muted text rather than a claim: the
/// platforms row says web is a text-only preview, the acceleration row keeps
/// LiteRT-LM's runtime capabilities separate from the backends litetune can
/// measure, and the models row says what has been measured end to end, which
/// is less than what exports. The last two are in the README's limitations
/// section and the first is in its opening; a compatibility table that quietly
/// drops any of them would be the exact failure the tool was built to catch.
///
/// The models row is cut to what was actually run, where cutting is what it
/// takes. The
/// qualifier names the two sizes and not "Gemma 3" because `models.py` scopes
/// `gemma-3-text` to the 270M and the 1B, and both are now measured, and
/// the list says "Gemma 3 (text)" for the same reason one level up: that
/// family rule covers those two sizes and no others. 4B and larger match no
/// rule at all. They are `Gemma3ForConditionalGeneration` with a vision
/// tower and a `model_type` of plain `gemma3`, which the exporter already
/// recognises — so the override the rule would add is unnecessary for them,
/// and it would assert a reason ("config.json says gemma3_text, which the
/// exporter does not recognise") that is untrue of them and would ship in the
/// manifest saying so. A family rule is a claim to have checked, and nobody
/// here has. They export
/// under the unknown-family note instead, so the row must not promise them.
///
/// Gemma 4 carries its variants for the same reason: `plan_export` returns an
/// unusable plan for a bare `gemma-4`, because the chat template override it
/// needs is per-variant and guessing one would ship a wrong template. E2B and
/// E4B pass, and so does a bare `gemma-4` whose caller supplies the override
/// itself — the refusal names that flag.
///
/// FunctionGemma is written bare because one size is all this repository has
/// ever named — `functiongemma-270m-it`, everywhere it appears. That is not
/// something `models.py` encodes: its rule matches the family generically,
/// unlike the Gemma 3 one, which enumerates. So if a second size ships, no
/// check here will notice, and this line needs a size too.
///
/// Qwen3 carries its sizes for the Gemma 3 reason: `models.py` claims `qwen-3`
/// by size, and 0.6B and 1.7B are the ones run. Its other sizes, and the Qwen 3
/// models built on other architectures, match no rule and export under the
/// unknown-family note. Qwen2.5 likewise: 0.5B and 1.5B Instruct.
///
/// The acceleration row names NPU only as a platform-specific LiteRT-LM
/// capability. `convert --npu` compiles for four Qualcomm SoCs (SM8550,
/// SM8650, SM8750, SM8850) on Linux x86-64 only, and for two kinds of model:
/// an EmbeddingGemma 2 graft's encoder, and a generation bundle for four
/// language-model families at the sizes that have run on an SM8750 NPU
/// (`gemma-3-text` 270M, `qwen-3` 0.6B, `qwen-2.5` 0.5B, `minicpm5` 1B; README,
/// *A generation bundle for a Qualcomm NPU*). Any other family or size is
/// refused, and `verify` refuses an NPU file, since LiteRT-LM's CPU and GPU
/// cannot run it. Keeping that out of the runtime sentence stops a reader
/// taking it as a promise about every bundle `convert` hands them.
///
/// "Conversion currently uses the CPU export path" holds with `--npu` too: its
/// export, calibration and compile run on a Linux x86-64 host, not on CUDA or
/// MPS.
class Formats extends StatelessComponent {
  const Formats({super.key});

  static const acceleration =
      'LiteRT-LM runs compatible bundles on CPU and GPU, with '
      'platform-specific NPU support';

  static const accelerationQualifier =
      ' — litetune can use CUDA or Apple GPU (MPS) for training and the float '
      'reference. Conversion currently uses the CPU export path; verify '
      'measures the converted model on CPU or GPU';

  /// Followed by a link to `/models`, which lists the measured runs one card
  /// apiece. The list used to be spelled out here and grew with every family.
  static const measuredQualifier =
      ' — fewer have been measured end to end than export; which, and how, is '
      'on ';

  /// The families `convert` exports. `/llms.txt` lists the same line.
  static const models =
      'Gemma 3 (text), Gemma 4 E2B/E4B, Qwen3 0.6B/1.7B, '
      'Qwen2.5 0.5B/1.5B Instruct, Qwen3.5, MiniCPM5 1B, Granite 4.1 3B, '
      'FunctionGemma, and EmbeddingGemma 2 as a decider';

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      h2(classes: 'label', [Component.text('Supported formats')]),
      div(classes: 'defs', [
        _row('Output', [
          span(classes: 'mono', [Component.text('.litertlm')]),
          Component.text(' — the format LiteRT-LM loads'),
        ]),
        _row('Platforms', [
          Component.text('Android, iOS, macOS, Linux and Windows natively'),
          span(classes: 'qualifier', [
            Component.text(
              ' — web runs as a text-only preview with no function calling and '
              'no LoRA, so a tuned tool-calling model is native-only for now',
            ),
          ]),
        ]),
        _row('Acceleration', [
          Component.text(acceleration),
          span(classes: 'qualifier', [Component.text(accelerationQualifier)]),
        ]),
        _row('Models', [
          Component.text(models),
          span(classes: 'qualifier', [
            Component.text(measuredQualifier),
            a(href: '/models', [Component.text('the models page')]),
          ]),
        ]),
      ]),
    ]);
  }

  static Component _row(String term, List<Component> definition) =>
      div(classes: 'def', [
        div(classes: 'def-term', [Component.text(term)]),
        div(classes: 'def-value', definition),
      ]);

  @css
  static List<StyleRule> get styles => [
    css('.defs').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      raw: const {'max-width': '62ch'},
      // A trailing rule under the last row closes the block; without it the
      // list reads as if it were cut off.
      border: Border.only(
        bottom: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.def').styles(
      display: Display.flex,
      gap: Gap.all(1.75.rem),
      padding: Padding.symmetric(vertical: 0.95.rem),
      border: Border.only(
        top: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.def-term').styles(
      color: Brand.muted,
      fontSize: 1.rem,
      width: 8.rem,
      flex: const Flex(shrink: 0),
    ),
    css(
      '.def-value',
    ).styles(color: Brand.ink, fontSize: 1.rem, lineHeight: 1.55.em),
    css('.qualifier').styles(color: Brand.muted),
    css('.qualifier a').styles(color: Brand.body),
    css('.mono').styles(fontFamily: Brand.fontMono, fontSize: 0.95.rem),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 640.px),
      styles: [
        css(
          '.def',
        ).styles(flexDirection: FlexDirection.column, gap: Gap.all(0.3.rem)),
        css('.def-term').styles(width: Unit.auto, fontSize: 0.9.rem),
      ],
    ),
  ];
}
