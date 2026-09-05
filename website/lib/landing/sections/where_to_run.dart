import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// Places a reader can take the artifact once litetune has produced it.
///
/// This exists because the pipeline's output is a file, and a file is not a
/// demo: without somewhere to load it, "runs on a phone" stays an assertion.
/// Both destinations accept a local `.litertlm` — one to build an app around,
/// one to try the file on a device without writing any code.
class WhereToRun extends StatelessComponent {
  const WhereToRun({super.key});

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      div(classes: 'label', [Component.text('Where to run it')]),
      div(classes: 'cards', [
        _card(
          href: 'https://fluttergemma.dev',
          name: 'flutter_gemma',
          note:
              'A Flutter plugin that runs .litertlm through the LiteRT-LM C '
              'API on five native platforms and a web preview. The same '
              'package also does embeddings and RAG.',
        ),
        _card(
          href: 'https://github.com/google-ai-edge/gallery',
          name: 'Google AI Edge Gallery',
          note:
              "Google's Android app for running local models on the device. "
              'Side-load the file to try it without writing an app first.',
        ),
      ]),
    ]);
  }

  static Component _card({
    required String href,
    required String name,
    required String note,
  }) => a(
    href: href,
    classes: 'card',
    attributes: const {'target': '_blank', 'rel': 'noopener'},
    [
      div(classes: 'card-name', [Component.text(name)]),
      div(classes: 'card-note', [Component.text(note)]),
    ],
  );

  @css
  static List<StyleRule> get styles => [
    css('.cards').styles(
      display: Display.flex,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(1.25.rem),
      raw: const {'max-width': '62ch'},
    ),
    css('.card').styles(
      // Grows to share the row, but never narrower than a readable card — so
      // the pair reflows to one column on a phone without a media query.
      flex: const Flex(grow: 1, shrink: 1, basis: Unit.pixels(240)),
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.6.rem),
      padding: Padding.all(1.4.rem),
      backgroundColor: Brand.surface,
      border: Border.all(color: Brand.line, width: 1.px),
      radius: BorderRadius.circular(0.6.rem),
      textDecoration: TextDecoration.none,
    ),
    css('.card:hover').styles(
      border: Border.all(color: Brand.muted, width: 1.px),
    ),
    css('.card-name').styles(
      fontFamily: Brand.fontMono,
      fontSize: 1.05.rem,
      letterSpacing: (-0.02).em,
      color: Brand.ink,
    ),
    css(
      '.card-note',
    ).styles(fontSize: 0.9.rem, lineHeight: 1.55.em, color: Brand.muted),
  ];
}
