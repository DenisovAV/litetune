import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../theme/brand.dart';
import 'sections/formats.dart';
import 'sections/hero.dart';
import 'sections/nav_bar.dart';
import 'sections/site_footer.dart';
import 'sections/what_it_does.dart';
import 'sections/where_to_run.dart';
import 'sections/why_it_exists.dart';

/// The whole site: one page, composed from sections that each live in their
/// own file under `lib/landing/sections/`.
class LandingPage extends StatelessComponent {
  const LandingPage({super.key});

  @override
  Component build(BuildContext context) {
    return div(classes: 'page', [
      const NavBar(),
      main_(classes: 'content', [
        const Hero(),
        const WhyItExists(),
        const Formats(),
        const WhereToRun(),
        const WhatItDoes(),
      ]),
      div(classes: 'content', [const SiteFooter()]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    ...Brand.rootStyles,
    css('*, *::before, *::after').styles(boxSizing: BoxSizing.borderBox),
    css('body').styles(
      margin: Margin.zero,
      padding: Padding.zero,
      fontFamily: Brand.fontSans,
      backgroundColor: Brand.bg,
      color: Brand.ink,
      raw: const {'-webkit-font-smoothing': 'antialiased'},
    ),
    css('.page').styles(minHeight: 100.vh, backgroundColor: Brand.bg),
    // One measure for every section, so the labels form a single column down
    // the left edge rather than drifting per section.
    css('.content').styles(
      maxWidth: 1120.px,
      margin: Margin.symmetric(horizontal: Unit.auto),
      padding: Padding.symmetric(horizontal: 2.rem),
    ),
    // Shared two-column section: a small uppercase label, then the content.
    // `.row` and `.label` are used by several sections; their look is defined
    // once here rather than repeated in each.
    css('.row').styles(
      display: Display.flex,
      gap: Gap.all(5.5.rem),
      alignItems: AlignItems.start,
      padding: Padding.only(top: 4.5.rem),
    ),
    css('.label').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.72.rem,
      letterSpacing: 0.12.em,
      textTransform: TextTransform.upperCase,
      color: Brand.muted,
      width: 11.rem,
      flex: const Flex(shrink: 0),
      lineHeight: 1.6.em,
    ),
    // Below the breakpoint the label stops being a gutter and becomes a
    // heading above its content — at phone width a fixed 11rem column would
    // leave the text a few characters wide.
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 900.px),
      styles: [
        css('.row').styles(
          flexDirection: FlexDirection.column,
          gap: Gap.all(1.1.rem),
          padding: Padding.only(top: 3.rem),
        ),
        css('.label').styles(width: Unit.auto),
      ],
    ),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 640.px),
      styles: [
        css(
          '.content',
        ).styles(padding: Padding.symmetric(horizontal: 1.25.rem)),
      ],
    ),
  ];
}
