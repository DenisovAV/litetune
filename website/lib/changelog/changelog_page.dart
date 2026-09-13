import 'dart:math' as math;

import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';
import 'package:markdown/markdown.dart' as md;

import '../landing/sections/nav_bar.dart';
import '../landing/sections/site_footer.dart';
import '../theme/brand.dart';

/// `/changelog`: `CHANGELOG.md`, rendered once at build time.
///
/// The page supplies its own title, so the file's `# Changelog` line is dropped
/// rather than rendered twice; everything after it -- the introduction and
/// every release -- is the file, unedited. Rendered here rather than linked to
/// on GitHub, because a release that makes an earlier result wrong says so at
/// the top of its entry, and that is the one thing a visitor to this site
/// should not have to leave it to find.
class ChangelogPage extends StatelessComponent {
  const ChangelogPage({
    required this.version,
    required this.markdown,
    super.key,
  });

  /// The package version, shown in the nav bar.
  final String version;

  /// The contents of `CHANGELOG.md`.
  final String markdown;

  /// `CHANGELOG.md` without its leading `# ` title, as HTML.
  ///
  /// Each release heading gets the id `v<version>`, so `/changelog#v0.1.5`
  /// links to that release. `gitHubFlavored` assigns headings no id at all;
  /// `gitHubWeb` would, but spelled from the whole heading -- `015--2026-09-08`
  /// -- which changes when a date is corrected.
  ///
  /// Refuses to render a file whose HTML is not what this page's own markdown
  /// writes -- markdown passes raw HTML in its input through untouched, and
  /// [keepOnlyTheTagsThisPageWrites] says what survives an allowlist of that.
  /// Refusing rather than quietly filtering: the page is a file in this
  /// repository, so markup in it is a mistake to fix, and a build that fails
  /// says so where a stripped page would not.
  ///
  /// What it cannot see is markup the allowlist would have written anyway: a
  /// hand-typed `<a href="https://...">` is byte-for-byte what a link in
  /// markdown becomes. That is a tidiness question, and
  /// `tests/test_changelog.py` is where it is asked; this is about what the
  /// page would carry.
  static String renderBody(String markdown) {
    final lines = markdown.split('\n');
    final start = lines.indexWhere((text) => text.trim().isNotEmpty);
    final body = start >= 0 && lines[start].startsWith('# ')
        ? lines.sublist(start + 1).join('\n')
        : markdown;
    final nodes = md.Document(
      extensionSet: md.ExtensionSet.gitHubFlavored,
    ).parse(body);
    for (final node in nodes) {
      if (node is md.Element && node.tag == 'h2') {
        final version = releaseHeading.firstMatch(node.textContent)?.group(1);
        if (version != null) node.attributes['id'] = 'v$version';
      }
    }
    final html = md.renderToHtml(nodes);
    final kept = keepOnlyTheTagsThisPageWrites(html);
    if (kept != html) {
      final at = _firstDifference(html, kept);
      throw StateError(
        'CHANGELOG.md renders HTML this page does not write, at: '
        '${html.substring(at, math.min(at + 80, html.length))} '
        '-- write it in backticks if it should read as text',      );
    }
    return html;
  }

  static int _firstDifference(String a, String b) {
    for (var i = 0; i < math.min(a.length, b.length); i++) {
      if (a[i] != b[i]) return i;
    }
    return math.min(a.length, b.length);
  }

  /// The tags this page's markdown writes, and their attributes.
  ///
  /// Wide enough that ordinary markdown does not fail a build: tables, task
  /// lists, images, headings, fence metadata and link titles are all here,
  /// because refusing a safe construct would turn an innocent changelog edit
  /// into a failed deploy. An `<img>` is held to this site, which is also
  /// what the site's `img-src` allows. What is left out is what a release
  /// entry has no business writing: footnote sections, and `<h1>`, which the
  /// page already has one of. Both are refused loudly rather than rendered.
  static const _allowedTags = <String, Set<String>>{
    'h2': {'id'},
    'h3': {},
    'h4': {},
    'h5': {},
    'h6': {},
    'p': {},
    'ul': {'class'},
    'ol': {'start', 'class'},
    'li': {'class'},
    // A task list's checkbox: markdown writes `type="checkbox"`, and
    // `checked="true"` when the box is ticked. `type` is held to `checkbox`
    // below -- a text field is not something a release entry writes.
    'input': {'type', 'checked'},
    'strong': {},
    'em': {},
    'del': {},
    // `data-metadata` is markdown's own, on the `<pre>` of a fence whose info
    // string has more than one word.
    'code': {'class'},
    'pre': {'data-metadata'},
    'blockquote': {},
    'hr': {},
    'br': {},
    'a': {'href', 'title'},
    'img': {'src', 'alt', 'title'},
    'table': {},
    'thead': {},
    'tbody': {},
    'tr': {},
    'th': {'align'},
    'td': {'align'},
  };

  /// The same HTML with every other tag shown as text and every link held to a
  /// plain scheme -- which [renderBody] uses to tell whether there were any.
  ///
  /// The page is `CHANGELOG.md` rendered, and markdown copies raw HTML from
  /// its input into its output -- so without this, a tag merged into that file
  /// would be a tag on the site. The site's Content-Security-Policy refuses a
  /// script there, but not everything a tag can do: a
  /// `<meta http-equiv="refresh">` sends the visitor elsewhere and a `<style>`
  /// or a positioned element covers the page, both measured against the built
  /// site under the real policy.
  ///
  /// An allowlist here rather than a rule about the file, because the rules
  /// are markdown's and hard to restate: a review got a `javascript:` link
  /// past a reader that stripped code spans by regex, since CommonMark closes
  /// a span only on a backtick run of the same length, and another past a
  /// reader that matched the destination as written, since markdown decodes
  /// `java&#x73;cript:` before it writes the `href`.
  ///
  /// A `<` that does not open a tag on this list is escaped where it stands,
  /// so an unterminated `<div` cannot leave one behind either. Links keep
  /// `href` and `title`, and the destination must be `http`, `https`,
  /// `mailto`, a place on this page or a path on this site.
  static String keepOnlyTheTagsThisPageWrites(String html) {
    final out = StringBuffer();
    var at = 0;
    while (at < html.length) {
      final next = html.indexOf('<', at);
      if (next < 0) {
        out.write(html.substring(at));
        break;
      }
      out.write(html.substring(at, next));
      final tag = _tag.matchAsPrefix(html, next);
      final name = tag?.group(2)?.toLowerCase();
      if (tag == null || !_allowedTags.containsKey(name)) {
        out.write('&lt;');
        at = next + 1;
        continue;
      }
      out.write(_rewritten(tag.group(1) == '/', name!, tag.group(3) ?? ''));
      at = tag.end;
    }
    return out.toString();
  }

  static final _tag = RegExp(r'<(/?)([A-Za-z][A-Za-z0-9-]*)([^>]*)>', dotAll: true);
  /// `key="value"`: markdown's renderer writes every attribute that way, a
  /// task list's `checked="true"` included. A bare one came from the file, and
  /// dropping it is what makes the difference that refuses the build.
  static final _attribute = RegExp(r'([A-Za-z-]+)\s*=\s*"([^"]*)"');
  static final _entity = RegExp(r'&(?:#(\d+)|#[xX]([0-9A-Fa-f]+)|([A-Za-z][A-Za-z0-9]*));');

  /// The classes markdown writes: a fence's language -- whatever the first
  /// word of its info string is, `c#` and `f*` included -- and a task list's
  /// two.
  static final _classes = RegExp(
    r'^(?:language-[^\s"]+|contains-task-list|task-list-item)$',
  );

  /// Characters that make a destination mean something other than it looks
  /// like: a backslash is a slash to a URL parser, so `\\host` and `/\host`
  /// are another origin, and a tab, a newline or a carriage return is dropped
  /// wherever it sits, so `/<tab>/host` is too. The rest below a space go at
  /// the ends only; they are refused with them rather than sorted.
  static final _slippery = RegExp(r'[\\\x00-\x1f]');

  /// A character reference this page would read differently from a browser.
  ///
  /// Markdown decodes a reference that ends in `;` and leaves one that does
  /// not; a browser decodes both. So `javascript&#58alert(1)` reached the page
  /// as a `javascript:` link past an earlier version of these checks, which
  /// read what markdown wrote. A numeric reference without its semicolon is
  /// refused here, and so is any named one with a semicolon but `&amp;` --
  /// rather than racing the browser's table of names. What is left, a numeric
  /// reference with its semicolon, is decoded below and read as a browser
  /// reads it: `&#39;` is the apostrophe markdown writes into an autolinked
  /// URL, and refusing that failed a build on ordinary prose.
  static final _reference = RegExp(
    r'&(?:#\d+(?![0-9;])|#[xX][0-9A-Fa-f]+(?![0-9A-Fa-f;])|(?!amp;)[A-Za-z][A-Za-z0-9]*;)',
  );

  static String _rewritten(bool closing, String name, String rest) {
    if (closing) return '</$name>';
    // An `<input>` with no `type` is a text field to a browser, and pinning
    // the value below cannot see one that was never written. The closing tag
    // above is left alone because markdown writes one -- `<input …></input>`
    // -- and a stray one on its own is nothing to a browser.
    if (name == 'input' && !rest.contains('type="checkbox"')) return '&lt;input$rest>';
    final kept = StringBuffer('<$name');
    for (final attribute in _attribute.allMatches(rest)) {
      final key = attribute.group(1)!.toLowerCase();
      if (!_allowedTags[name]!.contains(key)) continue;
      final value = attribute.group(2)!;
      if (key == 'href' && !_isPlainLink(value)) continue;
      if (key == 'src' && !_isOnThisSite(value)) continue;
      if (name == 'input' && key == 'type' && value != 'checkbox') continue;
      // Values, not just keys: these are markdown's own, and an attacker's
      // class would otherwise borrow whatever the site's stylesheet does with
      // one. Nothing here takes an element out of flow today; the first rule
      // that does would make an arbitrary class an overlay.
      if (key == 'class' && !_classes.hasMatch(value)) continue;
      if (key == 'align' && !const {'left', 'right', 'center'}.contains(value)) continue;
      kept.write(' $key="$value"');
    }
    kept.write(rest.trimRight().endsWith('/') ? ' />' : '>');
    return kept.toString();
  }

  /// A place on this site: what an `<img>` may load, since the site's policy
  /// serves images from here only and a remote one would be a broken image.
  /// Relative as well as rooted -- `assets/x.svg` is this site too.
  static bool _isOnThisSite(String value) {
    if (!_isPlainLink(value)) return false;
    final slashes = value.replaceAll(r'\', '/');
    return !RegExp(r'^[A-Za-z][A-Za-z0-9+.-]*:').hasMatch(slashes);
  }

  /// A destination a reader can follow safely: a page on this site, a place on
  /// this one, or plain mail. `//host/path` is not site-local -- a browser
  /// reads it as another origin with this page's scheme -- and a destination
  /// is refused outright when it carries a character reference this page would
  /// read differently ([_reference]) or a character a URL parser drops or
  /// re-reads ([_slippery]).
  static bool _isPlainLink(String value) {
    if (_reference.hasMatch(value) || _slippery.hasMatch(value)) return false;
    final decoded = value.replaceAllMapped(_entity, (m) {
      final code = m.group(1) != null
          ? int.tryParse(m.group(1)!)
          : m.group(2) != null
          ? int.tryParse(m.group(2)!, radix: 16)
          : const {'amp': 38, 'lt': 60, 'gt': 62, 'quot': 34, 'apos': 39}[m.group(3)];
      // Above the last code point there is nothing to decode to, and
      // `String.fromCharCode` throws rather than saying so; a browser reads
      // such a reference as the replacement character.
      return code == null || code > 0x10FFFF ? m.group(0)! : String.fromCharCode(code);
    }).trim();
    // Decoded as well as written: `&#9;` carries its semicolon, so it passes
    // the test above, and decodes to the tab a URL parser drops -- `/&#9;/host`
    // is `//host` to a browser.
    if (_slippery.hasMatch(decoded)) return false;
    if (decoded.startsWith('#')) return true;
    // A browser reads a backslash here as a slash, so `\\host`, `/\host` and
    // `//host` are all another origin with this page's scheme.
    final slashes = decoded.replaceAll(r'\', '/');
    if (slashes.startsWith('//')) return false;
    if (slashes.startsWith('/')) return true;
    final scheme = RegExp(r'^([A-Za-z][A-Za-z0-9+.-]*):').firstMatch(slashes);
    if (scheme == null) return !slashes.contains(':');
    return const {'http', 'https', 'mailto'}.contains(scheme.group(1)!.toLowerCase());
  }

  /// A release heading: a version -- an optional epoch, then numbers and dots,
  /// then anything else in the same word -- and whatever follows it (the
  /// date). Matched against the heading's text, after markdown has parsed
  /// it; `tests/test_changelog.py` reads the file to the same effect. A
  /// heading that does not start with a version gets no id and is not a
  /// release.
  static final releaseHeading = RegExp(r'^((?:\d+!)?\d+(?:\.\d+)*\S*)');

  @override
  Component build(BuildContext context) {
    return div(classes: 'page', [
      NavBar(version: version),
      main_(classes: 'content', [
        section(classes: 'changelog', [
          h1(classes: 'changelog-title', [Component.text('Changelog')]),
          // The leading newline is not layout. Jaspr's server renderer indents
          // raw HTML by rewriting every `\n` in it -- inside a `<pre>` too,
          // which put twelve spaces into a fenced command and into what a
          // reader copies from it. A first child that is whitespace with a
          // newline switches that off for its siblings: it is how jaspr
          // recognises parsed HTML (`markup_render_object.dart`, jaspr 0.23.4,
          // the "Special case" in `_renderChildren`).
          div(classes: 'changelog-body', [
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
    css('.changelog').styles(
      maxWidth: 46.rem,
      padding: Padding.only(top: 3.rem, bottom: 2.rem),
    ),
    css('.changelog-title').styles(
      margin: Margin.only(bottom: 1.5.rem),
      fontSize: 2.4.rem,
      fontWeight: FontWeight.w300,
      letterSpacing: (-0.02).em,
      color: Brand.ink,
    ),
    css('.changelog-body').styles(
      color: Brand.body,
      fontSize: 1.rem,
      lineHeight: 1.65.em,
    ),
    // One rule above each release, so the versions read as a column of entries
    // rather than one long list.
    css('.changelog-body h2').styles(
      margin: Margin.only(top: 3.rem, bottom: 1.rem),
      padding: Padding.only(top: 1.75.rem),
      border: Border.only(
        top: BorderSide(color: Brand.line, width: 1.px),
      ),
      fontFamily: Brand.fontMono,
      fontSize: 1.15.rem,
      fontWeight: FontWeight.w500,
      color: Brand.ink,
    ),
    css('.changelog-body ul').styles(
      margin: Margin.zero,
      padding: Padding.only(left: 1.25.rem),
    ),
    css('.changelog-body li').styles(margin: Margin.only(bottom: 0.6.rem)),
    css('.changelog-body strong').styles(
      color: Brand.ink,
      fontWeight: FontWeight.w500,
    ),
    css('.changelog-body code').styles(
      padding: Padding.symmetric(horizontal: 0.3.em, vertical: 0.1.em),
      backgroundColor: Brand.surface,
      fontFamily: Brand.fontMono,
      fontSize: 0.88.em,
      color: Brand.ink,
      raw: const {'border-radius': '3px', 'overflow-wrap': 'anywhere'},
    ),
    css('.changelog-body a').styles(
      color: Brand.ink,
      raw: const {'text-underline-offset': '3px'},
    ),
    StyleRule.media(
      query: MediaQuery.screen(maxWidth: 640.px),
      styles: [
        css('.changelog').styles(padding: Padding.only(top: 1.5.rem)),
        css('.changelog-title').styles(fontSize: 1.9.rem),
      ],
    ),
  ];
}
