import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// What comes out of the pipeline, and what it is compatible with.
///
/// Two rows carry a qualifier in muted text rather than a claim: the hardware
/// row says the measurement runs on CPU, and the models row says only one
/// family has been measured end to end. Both are in the README's limitations
/// section, and a compatibility table that quietly drops them would be the
/// exact failure the tool was built to catch.
///
/// The hardware row names NPU because the runtime has one — Snapdragon on
/// Android, Intel on Windows — while the qualifier keeps litetune's own claim
/// narrow: it measures on CPU. Naming a capability of the runtime and claiming
/// a measurement of it are different sentences; only the second would be
/// unsupported.
class Formats extends StatelessComponent {
  const Formats({super.key});

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      div(classes: 'label', [Component.text('Supported formats')]),
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
          Component.text(
            'CPU everywhere; GPU through OpenCL, Metal, Vulkan or DirectX 12; '
            'NPU on Snapdragon and Intel',
          ),
          span(classes: 'qualifier', [
            Component.text(
              ' — litetune measures on CPU, and every result records which '
              'backend produced it',
            ),
          ]),
        ]),
        _row('Models', [
          Component.text('Gemma 3, Gemma 4, Qwen 3.5, FunctionGemma'),
          span(classes: 'qualifier', [
            Component.text(
              ' — measured end to end on FunctionGemma and Gemma 3 270M so far',
            ),
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
