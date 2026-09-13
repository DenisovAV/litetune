import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// Wordmark and version on the left, links on the right. No call-to-action
/// button: the install line lives in the hero, and a second one up here would
/// compete with it for the same click.
class NavBar extends StatelessComponent {
  const NavBar({required this.version, super.key});

  /// The package version, from `src/litetune/_version.py` (see `project.dart`).
  /// Links to that release's changelog entry, which is where the next question
  /// about it leads.
  final String version;

  @override
  Component build(BuildContext context) {
    return header(classes: 'nav', [
      div(classes: 'nav-inner', [
        a(href: '/', classes: 'wordmark', [
          span(classes: 'wordmark-lite', [Component.text('lite')]),
          span(classes: 'wordmark-tune', [Component.text('tune')]),
        ]),
        a(href: '/changelog#v$version', classes: 'nav-version', [
          Component.text('v$version'),
        ]),
        nav(classes: 'nav-links', [
          a(href: '/changelog', [Component.text('Changelog')]),
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
    // Beside the wordmark rather than among the links: it is a fact about the
    // tool, not a destination, and it sits on the same baseline as the name.
    css('.nav-version').styles(
      padding: Padding.symmetric(horizontal: 0.45.rem, vertical: 0.15.rem),
      border: Border.all(color: Brand.line, width: 1.px),
      fontFamily: Brand.fontMono,
      fontSize: 0.75.rem,
      color: Brand.muted,
      textDecoration: TextDecoration.none,
      raw: const {'border-radius': '999px', 'white-space': 'nowrap'},
    ),
    css('.nav-version:hover').styles(color: Brand.ink),
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
        // Four links and a version do not fit one line at phone width; the
        // links wrap under the wordmark instead of overflowing the page.
        css('.nav-inner').styles(flexWrap: FlexWrap.wrap, gap: Gap.all(0.75.rem)),
        css('.nav-links').styles(
          flexWrap: FlexWrap.wrap,
          margin: Margin.only(left: Unit.zero),
          raw: const {'flex-basis': '100%'},
        ),
      ],
    ),
  ];
}
