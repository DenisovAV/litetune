import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// Wordmark on the left, three links on the right. No call-to-action button:
/// the install line lives in the hero, and a second one up here would compete
/// with it for the same click.
class NavBar extends StatelessComponent {
  const NavBar({super.key});

  @override
  Component build(BuildContext context) {
    return header(classes: 'nav', [
      div(classes: 'nav-inner', [
        a(href: '/', classes: 'wordmark', [
          span(classes: 'wordmark-lite', [Component.text('lite')]),
          span(classes: 'wordmark-tune', [Component.text('tune')]),
        ]),
        nav(classes: 'nav-links', [
          _link('https://github.com/DenisovAV/litetune', 'GitHub'),
          _link('https://pypi.org/project/litetune/', 'PyPI'),
          _link(
            'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md',
            'Measurements',
          ),
        ]),
      ]),
    ]);
  }

  static Component _link(String href, String label) => a(
    href: href,
    attributes: const {'target': '_blank', 'rel': 'noopener'},
    [Component.text(label)],
  );

  @css
  static List<StyleRule> get styles => [
    css('.nav').styles(
      padding: Padding.symmetric(vertical: 1.75.rem, horizontal: 2.rem),
    ),
    css('.nav-inner').styles(
      display: Display.flex,
      alignItems: AlignItems.center,
      gap: Gap.all(1.5.rem),
      maxWidth: 1120.px,
      margin: Margin.symmetric(horizontal: Unit.auto),
    ),
    // The wordmark is the logo: Plex Mono, the two halves split by weight, with
    // the negative tracking the lockup was drawn at.
    css('.wordmark').styles(
      fontFamily: Brand.fontMono,
      fontSize: 1.3.rem,
      letterSpacing: (-0.04).em,
      lineHeight: 1.em,
      color: Brand.ink,
      textDecoration: TextDecoration.none,
    ),
    css('.wordmark-lite').styles(fontWeight: FontWeight.w300),
    css('.wordmark-tune').styles(fontWeight: FontWeight.w600),
    css('.nav-links').styles(
      display: Display.flex,
      gap: Gap.all(1.75.rem),
      margin: Margin.only(left: Unit.auto),
      fontFamily: Brand.fontMono,
      fontSize: 0.85.rem,
    ),
    css(
      '.nav-links a',
    ).styles(color: Brand.muted, textDecoration: TextDecoration.none),
    css('.nav-links a:hover').styles(color: Brand.ink),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 640.px),
      styles: [
        css('.nav').styles(
          padding: Padding.symmetric(vertical: 1.25.rem, horizontal: 1.25.rem),
        ),
        css('.nav-links').styles(gap: Gap.all(1.1.rem), fontSize: 0.8.rem),
      ],
    ),
  ];
}
