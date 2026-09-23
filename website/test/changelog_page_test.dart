// What `ChangelogPage.renderBody` refuses, and what it must go on rendering.
//
// Every entry below was produced by a review round against an earlier version
// of this check that let it through, so each one is a regression test rather
// than an invention. The page is `CHANGELOG.md` rendered as HTML, so markup in
// that file would be markup on the site: the check exists to make the build
// fail instead.
import 'package:litetune_website/changelog/changelog_page.dart';
import 'package:litetune_website/project.dart';
import 'package:test/test.dart';

/// A changelog with one ordinary entry, then whatever is being tested.
String changelog(String body) =>
    '# Changelog\n\n## 9.0.0 — 2026-09-13\n\n- An entry.\n\n$body\n';

void main() {
  group('refuses', () {
    const refused = <String, String>{
      'a tag':
          '<meta http-equiv="refresh" content="0;url=https://evil.example">',
      'a tag split over lines':
          '<style\ntype="text/css">.changelog-body{display:none}</style\n>',
      'a tag left unterminated':
          '<div style="position:fixed;inset:0;background:red"',
      'a fence shadowed by a tag':
          '<div\n```\n<meta http-equiv="refresh" content="0;url=https://e.example/">\n```',
      'a tag inside a code span that markdown does not close':
          '- Fixed the ``` <style>body{display:none}</style>` `` fence handling.',
      'a tag between two stray backticks':
          'A note with a stray ` backtick.\n\n<meta http-equiv="refresh" content="3;url=https://e.example">\n\nAnother stray ` backtick.',
      'a javascript: link': '- [notes](javascript:alert(1))',
      'a javascript: link written with an entity':
          '- [notes](&#106;avascript:alert(1))',
      'a javascript: link whose entity has no semicolon':
          '- [notes](javascript&#58alert(1))',
      'a javascript: link whose entity is hexadecimal':
          '- [notes](javascript&#x3A+alert(1))',
      'a javascript: link written with a named entity':
          '- [notes](javascript&colon;alert(1))',
      'a javascript: link in an angle-bracketed destination':
          '- [notes](<&#106;avascript:alert(1)>)',
      'a javascript: link through a reference':
          '- [notes][r]\n\n[r]: &#106;avascript:alert(1)',
      'a javascript: autolink': '- <javascript:alert(1)>',
      'a scheme-relative link': '- [notes](//evil.example/path)',
      'a scheme-relative link spelled with backslashes':
          '<a href="\\\\evil.example/path">notes</a>',
      'a link whose slash hides a tab':
          '<a href="/\t/evil.example/path">notes</a>',
      'a remote image': '- ![chart](https://evil.example/x.png)',
      'a data: image': '- ![chart](data:image/svg+xml;base64,AAA)',
      'an ftp: autolink': '- <ftp://example.test/file>',
      'a heading the page already has': '# Another title',
      'a text input': '<input type="text">',
      'an input with no type at all': '<input>',
      'a link whose control character is written as an entity':
          '<a href="&#1;//evil.example/">notes</a>',
      'a javascript: link written with a closed hexadecimal entity':
          '<a href="java&#x73;cript:alert(1)">notes</a>',
      'an event handler': '<a href="/x" onclick="alert(1)">notes</a>',
      'a footnote': 'A note with one.[^1]\n\n[^1]: The note.',
      'an image whose path hides a tab':
          '<img src="/\t/evil.example/p.png" alt="x">',
      'a link whose tab is written as an entity':
          '<a href="/&#9;/evil.example/">notes</a>',
      'a link whose newline is written as an entity':
          '<a href="/&#10;/evil.example/">notes</a>',
      'an attribute written without a value':
          '<input type="checkbox" disabled>',
      'a class the page does not write': '<ul class="nav"><li>x</li></ul>',
      'an alignment that is not one':
          '<table><tr><td align="javascript:">x</td></tr></table>',
      'a placeholder written as a tag':
          'Pass --output <dir> to choose where it goes.',
    };

    refused.forEach((what, body) {
      test(what, () {
        expect(
          () => ChangelogPage.renderBody(changelog(body)),
          throwsA(isA<StateError>()),
          reason: '$what reached the page',
        );
      });
    });
  });

  group('renders', () {
    const kept = <String, String>{
      'prose, emphasis and code spans':
          '- **bold**, _quiet_, `<pad>` and ~~gone~~.',
      'a link': '- [the notes](https://github.com/DenisovAV/litetune/releases)',
      'a link with a title':
          '- [the notes](https://litetune.dev "the changelog")',
      'a link with a query string':
          '- [issues](https://github.com/x/y?q=is%3Aopen&sort=created)',
      'a URL with an apostrophe':
          "- See https://en.wikipedia.org/wiki/Occam's_razor for details.",
      'a link to this site':
          '- [the entry](/changelog#v9.0.0) and [here](#v9.0.0)',
      'an image on this site': '- ![chart](/images/chart.png "a chart")',
      'an image by a relative path': '- ![diagram](assets/diagram.svg)',
      'an autolink': '- <https://litetune.dev/changelog>',
      'an email autolink': '- <sasha@example.com>',
      'a list, nested': '- one\n  - two\n- three',
      'an ordered list': '3. three\n4. four',
      'a task list': '- [x] done\n- [ ] not yet',
      'an ordered task list': '1. [ ] todo',
      'a table': '| Before | After |\n| --- | ---: |\n| slow | fast |',
      'a fence with a language': '```bash\nlitetune verify\n```',
      'a fence with a longer info string':
          '```bash on macOS\nlitetune verify\n```',
      'a fence whose language has punctuation': '```c#\nvar x = 1;\n```',
      'a quote, a rule and a sub-heading': '> quoted\n\n---\n\n### deeper',
      'a less-than in prose': '- values < 5 are dropped',
    };

    kept.forEach((what, body) {
      test(what, () {
        expect(
          () => ChangelogPage.renderBody(changelog(body)),
          returnsNormally,
          reason: what,
        );
      });
    });

    test('a table as a table', () {
      final html = ChangelogPage.renderBody(
        changelog('| a | b |\n| --- | --- |\n| c | d |'),
      );
      expect(html, contains('<table>'));
      expect(html, contains('<td>c</td>'));
    });

    test('a task list as checkboxes', () {
      final html = ChangelogPage.renderBody(
        changelog('- [x] done\n- [ ] not yet'),
      );
      expect(html, contains('class="contains-task-list"'));
      expect(html, contains('<input type="checkbox" checked="true">'));
      expect(html, contains('<input type="checkbox">'));
    });

    test('the release heading, with an id to link to', () {
      final html = ChangelogPage.renderBody(changelog('- An entry.'));
      expect(html, contains('<h2 id="v9.0.0">9.0.0 — 2026-09-13</h2>'));
    });

    test("the repository's own changelog", () {
      final html = ChangelogPage.renderBody(changelogMarkdown);
      expect(html, contains('<h2 id="v0.1.5">'));
    });
  });
}
