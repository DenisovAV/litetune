import 'dart:math' as math;

import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';
import 'package:markdown/markdown.dart' as md;

import '../changelog/changelog_page.dart';
import '../landing/sections/nav_bar.dart';
import '../landing/sections/site_footer.dart';
import '../theme/brand.dart';

/// `/measurements`: `MEASUREMENTS.md`, rendered once at build time.
///
/// Rendered here rather than linked to on GitHub for the reason the cards
/// exist at all: a card carries no score, so the reader who wants one is sent
/// to the table with its interval, its sample size and its refusals beside it,
/// and sending them off the site to read it made the numbers feel like
/// somebody else's.
///
/// It also closes a drift this site used to be exposed to. A card named the
/// section it opened as a hand-written anchor, and nothing but a test could
/// tell whether a heading in another file still spelled it that way. The page
/// and the link now come out of one parse of one file: [headingId] spells an
/// id, [renderBody] puts it on the heading, and the card's href is the same
/// string. A section that is renamed takes its link with it.
class MeasurementsPage extends StatelessComponent {
  const MeasurementsPage({
    required this.version,
    required this.markdown,
    super.key,
  });

  /// The package version, shown in the nav bar.
  final String version;

  /// The contents of `MEASUREMENTS.md`.
  final String markdown;

  /// GitHub's own anchor for a heading: lowercased, punctuation dropped,
  /// spaces hyphenated.
  ///
  /// Spelled GitHub's way rather than this site's own way because the file is
  /// read in both places -- it is `MEASUREMENTS.md` in the repository as well
  /// as a page here -- and a reader who follows a link from a README to
  /// GitHub and a reader who follows a card to this page should land on the
  /// same section.
  static String headingId(String heading) => heading
      .toLowerCase()
      .replaceAll(RegExp(r'[^\w\- ]'), '')
      .replaceAll(' ', '-');

  /// Every `##` and `###` heading this file has, in order, as ids.
  ///
  /// What a card's link is checked against: the ids that exist are the ids the
  /// page emits, not the ones a heading in a file might spell.
  static List<String> idsIn(String markdown) {
    final nodes = md.Document(
      extensionSet: md.ExtensionSet.gitHubFlavored,
    ).parse(_withoutTitle(markdown));
    return [
      for (final node in nodes)
        if (node is md.Element && (node.tag == 'h2' || node.tag == 'h3'))
          headingId(node.textContent),
    ];
  }

  /// The file after its leading `# ` line, which the page's own title replaces.
  static String _withoutTitle(String markdown) {
    final lines = markdown.split('\n');
    final start = lines.indexWhere((text) => text.trim().isNotEmpty);
    return start >= 0 && lines[start].startsWith('# ')
        ? lines.sublist(start + 1).join('\n')
        : markdown;
  }

  /// `MEASUREMENTS.md` without its leading `# ` title, as HTML.
  ///
  /// Every `##` and `###` gets an id, so a card can open the section that
  /// measured its model. Held to the same allowlist the changelog is held to,
  /// and refused the same way: markdown passes raw HTML in its input straight
  /// through, and this file is one a pull request edits.
  static String renderBody(String markdown) {
    final nodes = md.Document(
      extensionSet: md.ExtensionSet.gitHubFlavored,
    ).parse(_withoutTitle(markdown));
    for (final node in nodes) {
      if (node is md.Element && (node.tag == 'h2' || node.tag == 'h3')) {
        node.attributes['id'] = headingId(node.textContent);
      }
    }
    final html = md.renderToHtml(nodes);
    final kept = ChangelogPage.keepOnlyTheTagsThisPageWrites(html);
    if (kept != html) {
      final at = _firstDifference(html, kept);
      throw StateError(
        'MEASUREMENTS.md renders HTML this page does not write, at: '
        '${html.substring(at, math.min(at + 80, html.length))} '
        '-- write it in backticks if it should read as text',
      );
    }
    return html;
  }

  static int _firstDifference(String a, String b) {
    for (var i = 0; i < math.min(a.length, b.length); i++) {
      if (a[i] != b[i]) return i;
    }
    return math.min(a.length, b.length);
  }

  @override
  Component build(BuildContext context) {
    return div(classes: 'page', [
      NavBar(version: version),
      main_(classes: 'content', [
        section(classes: 'measurements', [
          h1(classes: 'measurements-title', [
            Component.text('What was measured, and what it established'),
          ]),
          // See `ChangelogPage.build`: a first child that is whitespace with a
          // newline is how jaspr recognises parsed HTML and stops indenting
          // it, which otherwise rewrites every newline inside a `<pre>`.
          div(classes: 'measurements-body', [
            Component.text('\n'),
            RawText(renderBody(markdown)),
          ]),
        ]),
      ]),
      div(classes: 'content', [const SiteFooter()]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.measurements').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.5.rem),
      padding: Padding.symmetric(vertical: 3.rem),
    ),
    css('.measurements-title').styles(
      fontSize: 1.9.rem,
      fontWeight: FontWeight.w600,
      color: Brand.ink,
      raw: const {'line-height': '1.2'},
    ),
    // Wider than the 62ch prose column the landing page uses: these are
    // tables, and that measure wraps every one of them into unreadability.
    css('.measurements-body').styles(
      color: Brand.body,
      fontSize: 1.rem,
      lineHeight: 1.65.em,
      raw: const {'max-width': '80ch'},
    ),
    css('.measurements-body h2').styles(
      fontFamily: Brand.fontMono,
      fontSize: 1.2.rem,
      fontWeight: FontWeight.w500,
      color: Brand.ink,
      border: Border.only(
        top: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
      padding: Padding.only(top: 1.6.rem),
      margin: Margin.only(top: 2.4.rem, bottom: 0.6.rem),
    ),
    css('.measurements-body h3').styles(
      fontSize: 1.05.rem,
      fontWeight: FontWeight.w600,
      color: Brand.ink,
      margin: Margin.only(top: 1.8.rem, bottom: 0.5.rem),
    ),
    css(
      '.measurements-body strong',
    ).styles(color: Brand.ink, fontWeight: FontWeight.w500),
    css('.measurements-body code').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.85.em,
      color: Brand.ink,
      backgroundColor: Brand.surface,
      padding: Padding.symmetric(horizontal: 0.3.rem, vertical: 0.1.rem),
    ),
    css(
      '.measurements-body a',
    ).styles(color: Brand.ink, raw: const {'text-underline-offset': '3px'}),
    // A table here is the artifact, not decoration: it scrolls on its own
    // rather than shrinking the type, because a cell reading `+0.0217 ±0.0142`
    // is unreadable wrapped and meaningless truncated.
    css('.measurements-body table').styles(
      fontSize: 0.85.rem,
      margin: Margin.symmetric(vertical: 1.rem),
      raw: const {
        'border-collapse': 'collapse',
        'display': 'block',
        'overflow-x': 'auto',
        'max-width': '100%',
      },
    ),
    css('.measurements-body th').styles(
      color: Brand.ink,
      fontWeight: FontWeight.w500,
      padding: Padding.symmetric(horizontal: 0.6.rem, vertical: 0.4.rem),
      raw: const {
        'border': '1px solid var(--lt-line)',
        'text-align': 'left',
        'white-space': 'nowrap',
      },
    ),
    css('.measurements-body td').styles(
      padding: Padding.symmetric(horizontal: 0.6.rem, vertical: 0.4.rem),
      raw: const {
        'border': '1px solid var(--lt-line)',
        'text-align': 'left',
        'white-space': 'nowrap',
      },
    ),
    css('.measurements-body blockquote').styles(
      color: Brand.muted,
      border: Border.only(
        left: BorderSide(
          color: Brand.line,
          width: 2.px,
          style: BorderStyle.solid,
        ),
      ),
      padding: Padding.only(left: 1.rem),
      margin: Margin.symmetric(vertical: 1.rem),
    ),
    css.media(MediaQuery.screen(maxWidth: 640.px), [
      css('.measurements').styles(padding: Padding.only(top: 1.5.rem)),
      css('.measurements-title').styles(fontSize: 1.5.rem),
    ]),
  ];
}
