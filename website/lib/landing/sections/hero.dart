import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';
import 'runtimes.dart';

/// The first screen: one descriptive sentence, the runtimes, how to install it.
///
/// The headline names no file format and no runtime on purpose — those are
/// proper nouns a reader either knows or does not, and a sentence that opens
/// with one loses everybody in the second group. They follow immediately, in
/// blocks that are labelled and skimmable.
class Hero extends StatelessComponent {
  const Hero({super.key});

  @override
  Component build(BuildContext context) {
    return section(classes: 'hero', [
      h1(classes: 'hero-h1', [
        Component.text(
          'litetune fine-tunes a small language model, converts it into a file '
          'that runs on a phone, and checks that the converted model still does '
          'the job.',
        ),
      ]),
      const RuntimesStrip(),
      div(classes: 'hero-install-row', [
        div(classes: 'install', [
          span(classes: 'install-prompt', [Component.text(r'$')]),
          span(classes: 'install-cmd', [
            Component.text('pip install litetune'),
          ]),
        ]),
        div(classes: 'install', [
          span(classes: 'install-prompt', [Component.text(r'$')]),
          span(classes: 'install-cmd', [
            Component.text('brew install DenisovAV/tap/litetune'),
          ]),
        ]),
      ]),
      p(classes: 'hero-meta', [
        Component.text(
          'Apache-2.0 · Python 3.10–3.12 · macOS and Linux · alpha',
        ),
      ]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.hero').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(2.25.rem),
      padding: Padding.only(top: 4.rem, bottom: 1.rem),
    ),
    // Light weight at a large size: the sentence is long, and at 400 it would
    // read as a wall rather than as an opening line.
    css('.hero-h1').styles(
      fontFamily: Brand.fontSans,
      fontSize: 2.5.rem,
      fontWeight: FontWeight.w300,
      letterSpacing: (-0.022).em,
      lineHeight: 1.4.em,
      color: Brand.ink,
      raw: const {'max-width': '60ch'},
      margin: Margin.zero,
    ),
    css('.hero-install-row').styles(
      display: Display.flex,
      alignItems: AlignItems.center,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(1.75.rem),
    ),
    // Both install methods get the same box. An earlier revision boxed pip and
    // left Homebrew as plain text beside it, meaning it as primary-vs-secondary;
    // it read as two unrelated elements instead. Neither is more correct than
    // the other, so neither is styled as if it were.
    css('.install').styles(
      display: Display.inlineFlex,
      alignItems: AlignItems.center,
      fontFamily: Brand.fontMono,
      fontSize: 0.95.rem,
      color: Brand.ink,
      padding: Padding.symmetric(vertical: 0.9.rem, horizontal: 1.35.rem),
      border: Border.all(color: Brand.line, width: 1.px),
      radius: BorderRadius.circular(0.5.rem),
    ),
    // The prompt is decoration, not part of the command: excluding it from
    // selection means a drag across the box copies something runnable.
    // A leading space inside the command span collapses under HTML whitespace
    // rules and rendered as `\$pip`. The gap is a margin so it cannot be
    // swallowed, and the prompt stays out of a selection so that dragging
    // across the box copies something runnable.
    css('.install-prompt').styles(
      color: Brand.muted,
      userSelect: UserSelect.none,
      margin: Margin.only(right: 0.5.rem),
    ),
    css('.hero-meta').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.8.rem,
      color: Brand.muted,
      margin: Margin.zero,
    ),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 768.px),
      styles: [
        css('.hero').styles(
          padding: Padding.only(top: 2.5.rem, bottom: 0.5.rem),
        ),
        css('.hero-h1').styles(fontSize: 1.65.rem, lineHeight: 1.42.em),
      ],
    ),
  ];
}
