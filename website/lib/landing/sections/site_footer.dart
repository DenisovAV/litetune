import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

class SiteFooter extends StatelessComponent {
  const SiteFooter({super.key});

  @override
  Component build(BuildContext context) {
    return footer(classes: 'foot', [
      div(classes: 'foot-left', [
        div(classes: 'wordmark foot-wordmark', [
          span(classes: 'wordmark-lite', [Component.text('lite')]),
          span(classes: 'wordmark-tune', [Component.text('tune')]),
        ]),
        div(classes: 'foot-credit', [
          a(
            href: 'https://sashadenisov.dev',
            attributes: const {'target': '_blank', 'rel': 'noopener'},
            [Component.text('Sasha Denisov')],
          ),
          Component.text(' · Apache-2.0'),
        ]),
      ]),
      nav(classes: 'nav-links foot-links', [
        _link('https://github.com/DenisovAV/litetune', 'GitHub'),
        _link('https://pypi.org/project/litetune/', 'PyPI'),
        _link(
          'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md',
          'Measurements',
        ),
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
    css('.foot').styles(
      display: Display.flex,
      alignItems: AlignItems.end,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(2.rem),
      padding: Padding.only(top: 2.25.rem, bottom: 3.5.rem),
      border: Border.only(
        top: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.foot-left').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.6.rem),
    ),
    css('.foot-wordmark').styles(fontSize: 1.15.rem),
    css('.foot-credit').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.78.rem,
      color: Brand.muted,
    ),
    // The name is the only link in the footer that is not a project resource,
    // so it is underlined rather than left to look like plain text.
    css('.foot-credit a').styles(
      color: Brand.muted,
      raw: const {
        'text-decoration': 'underline',
        'text-underline-offset': '2px',
      },
    ),
    css('.foot-credit a:hover').styles(color: Brand.ink),
    css('.foot-links').styles(margin: Margin.only(left: Unit.auto)),
  ];
}
