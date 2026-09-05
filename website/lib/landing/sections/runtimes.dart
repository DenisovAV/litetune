import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The runtimes that can load what litetune produces.
///
/// Rendered inside the hero rather than as a section of its own: for the
/// audience this page is written for, "which runtime" is part of deciding
/// whether the tool is relevant at all, so it sits above the install line.
///
/// There is exactly one entry, and that is the honest state of things — the
/// heading says `SUPPORTED RUNTIMES` in the plural so that a second one can
/// join it without the label becoming a lie. The `flutter_gemma` plugin is
/// deliberately NOT listed here: it embeds LiteRT-LM rather than being a
/// runtime, and it appears under formats instead.
class RuntimesStrip extends StatelessComponent {
  const RuntimesStrip({super.key});

  @override
  Component build(BuildContext context) {
    return div(classes: 'runtimes', [
      div(classes: 'label', [Component.text('Supported runtimes')]),
      div(classes: 'runtimes-list', [
        div(classes: 'runtime', [
          div(classes: 'runtime-name', [Component.text('LiteRT-LM')]),
          div(classes: 'runtime-note', [
            Component.text("Google's on-device LLM runtime"),
          ]),
        ]),
      ]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.runtimes').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.25.rem),
      padding: Padding.symmetric(vertical: 1.75.rem),
      border: Border.symmetric(
        vertical: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.runtimes-list').styles(
      display: Display.flex,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(4.5.rem),
    ),
    css('.runtime').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.55.rem),
    ),
    css('.runtime-name').styles(
      fontFamily: Brand.fontMono,
      fontSize: 1.9.rem,
      fontWeight: FontWeight.w400,
      letterSpacing: (-0.035).em,
      lineHeight: 1.em,
      color: Brand.ink,
    ),
    css(
      '.runtime-note',
    ).styles(fontSize: 0.9.rem, color: Brand.muted, lineHeight: 1.5.em),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 640.px),
      styles: [css('.runtime-name').styles(fontSize: 1.5.rem)],
    ),
  ];
}
