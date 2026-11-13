// What search engines and language models read: the social card, the titles
// and descriptions, the headings, the JSON-LD and `/llms.txt`.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:jaspr/server.dart';
import 'package:litetune_website/changelog/changelog_page.dart';
import 'package:litetune_website/landing/landing_page.dart';
import 'package:litetune_website/landing/sections/quick_start.dart';
import 'package:litetune_website/landing/sections/where_to_run.dart';
import 'package:litetune_website/llms_txt.dart';
import 'package:litetune_website/measurements/measurements_page.dart';
import 'package:litetune_website/models/embedding_models.dart';
import 'package:litetune_website/models/llm_models.dart';
import 'package:litetune_website/models/models_page.dart';
import 'package:litetune_website/seo.dart';
import 'package:test/test.dart';

Directory _root() {
  for (final candidate in [Directory.current.parent, Directory.current]) {
    if (File('${candidate.path}/README.md').existsSync()) return candidate;
  }
  throw StateError('cannot find README.md from ${Directory.current.path}');
}

/// The `<tag` openings in [html], `<h1` matching `<h1>` and `<h1 class=...>`
/// but not `<h10`.
int _count(String html, String tag) =>
    RegExp('<$tag[\\s>]').allMatches(html).length;

Future<String> _render(Component page) async {
  final response = await renderComponent(page, standalone: true);
  return utf8.decode(response.body);
}

/// Every pre-rendered route, as `main.server.dart` registers them.
final _routes = {
  '/',
  ModelsTab.llm.path,
  ModelsTab.embeddings.path,
  '/changelog',
  '/measurements',
  kLlmsTxtPath,
  kLlmsFullTxtPath,
};

void main() {
  final root = _root();
  final readme = File('${root.path}/README.md').readAsStringSync();
  final measurements = File('${root.path}/MEASUREMENTS.md').readAsStringSync();
  final changelog = File('${root.path}/CHANGELOG.md').readAsStringSync();

  setUpAll(Jaspr.initializeApp);

  group('the social card', () {
    // `test` runs from website/, which is where `web/` is.
    final file = File('web$kOgImagePath');

    test('is a PNG of the size the head declares', () {
      expect(file.existsSync(), isTrue, reason: '${file.path} is missing');
      final bytes = file.readAsBytesSync();
      expect(bytes.sublist(0, 8), [
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,
      ], reason: 'not a PNG signature');
      // IHDR is the first chunk: width and height are big-endian at 16 and 20.
      final header = ByteData.sublistView(bytes, 16, 24);
      expect(header.getUint32(0), kOgImageWidth);
      expect(header.getUint32(4), kOgImageHeight);
      expect((kOgImageWidth, kOgImageHeight), (1200, 630));
    });

    test('is named in the head with its size and alt text', () async {
      final html = await _render(
        Component.fragment(seoHead(title: 't', description: 'd')),
      );
      expect(html, contains('content="$kSiteOrigin$kOgImagePath"'));
      for (final property in [
        'og:image:width" content="1200"',
        'og:image:height" content="630"',
        'og:image:alt"',
        'twitter:image:alt"',
      ]) {
        expect(html, contains(property));
      }
    });
  });

  group('titles and descriptions', () {
    test('the landing page names what people search for', () {
      for (final word in ['fine-tune', 'Gemma', 'Qwen', 'LiteRT-LM']) {
        expect(kLandingTitle, contains(word));
      }
      for (final word in [
        'full or LoRA',
        'Gemma 3',
        'Gemma 4',
        'Qwen',
        'FunctionGemma',
        'EmbeddingGemma 2',
        '.litertlm',
        'on-device',
        'quantization',
        'float model',
      ]) {
        expect(kLandingDescription, contains(word));
      }
      // `tune` defaults to a full fine-tune; LoRA is never the only method.
      expect(kLandingDescription, isNot(contains('with LoRA')));
    });

    test('the models pages name every family on their cards', () {
      for (final run in LlmModels.measured) {
        expect(kModelsDescription, contains(run.name.split(' ').first));
      }
      for (final run in EmbeddingModels.measured) {
        expect(kEmbeddingsDescription, contains(run.name));
      }
      expect(kModelsDescription, contains('Gemma 3,'));
      expect(kModelsDescription, isNot(contains('270M')));
    });
  });

  group('headings', () {
    test('the landing page has one h1 and its sections are h2s', () async {
      final html = await _render(const LandingPage(version: '0.0.0'));
      expect(_count(html, 'h1'), 1);
      expect(_count(html, 'h2'), greaterThanOrEqualTo(6));
      expect(_count(html, 'div class="label"'), 0);
      for (final heading in [
        'Supported runtimes',
        'Why it exists',
        'Supported formats',
        'Where to run it',
        'What it does',
        QuickStart.heading,
      ]) {
        expect(html, contains('<h2 class="label">$heading</h2>'));
      }
    });

    for (final tab in ModelsTab.values) {
      test('${tab.path} has one h1 and an h2', () async {
        final html = await _render(ModelsPage(version: '0.0.0', tab: tab));
        expect(_count(html, 'h1'), 1);
        expect(_count(html, 'h2'), greaterThanOrEqualTo(1));
      });
    }

    test('/changelog and /measurements have one h1 each', () async {
      final pages = [
        ChangelogPage(version: '0.0.0', markdown: changelog),
        MeasurementsPage(version: '0.0.0', markdown: measurements),
      ];
      for (final page in pages) {
        expect(_count(await _render(page), 'h1'), 1);
      }
    });
  });

  group('the quick start', () {
    test('quotes the README rather than paraphrasing it', () {
      final lines = readme.split('\n');
      for (final line in QuickStart.commands) {
        if (line.isEmpty) continue;
        if (line.startsWith('#')) {
          expect(
            lines.any((l) => l.startsWith(line)),
            isTrue,
            reason: 'no README line begins "$line"',
          );
        } else {
          expect(lines, contains(line), reason: 'not a README line: "$line"');
        }
      }
    });

    test('runs prepare, tune, convert and verify in that order', () {
      final order = [
        for (final line in QuickStart.commands)
          if (line.startsWith('litetune ')) line.split(' ')[1],
      ];
      expect(order, ['prepare', 'tune', 'convert', 'verify']);
    });

    test('links a README section that exists', () {
      final anchor = QuickStart.readmeUrl.split('#').last;
      expect(readme, contains('\n## From your data to a shippable bundle\n'));
      expect(anchor, 'from-your-data-to-a-shippable-bundle');
    });
  });

  group('JSON-LD', () {
    test('names the version, the source and the package', () {
      final json = applicationJsonLd(description: 'd', version: '1.2.3');
      expect(json['@type'], ['SoftwareApplication', 'SoftwareSourceCode']);
      expect(json['softwareVersion'], '1.2.3');
      expect(json['codeRepository'], 'https://github.com/DenisovAV/litetune');
      expect(json['downloadUrl'], 'https://pypi.org/project/litetune/');
      expect(json['sameAs'], [kRepositoryUrl, kPypiUrl]);
      expect(json['programmingLanguage'], 'Python');
      expect(json['keywords'], allOf(contains('LoRA'), contains('LiteRT-LM')));
      expect(
        applicationJsonLd(description: 'd').containsKey('softwareVersion'),
        isFalse,
      );
    });

    test('is in the head, with the version passed', () async {
      final html = await _render(
        Component.fragment(
          seoHead(title: 't', description: 'd', version: '1.2.3'),
        ),
      );
      final body = RegExp(
        r'<script type="application/ld\+json">(.*?)</script>',
        dotAll: true,
      ).firstMatch(html)!.group(1)!;
      expect((jsonDecode(body) as Map)['softwareVersion'], '1.2.3');
    });
  });

  test('flutter_gemma is linked where it lives, not through a redirect', () {
    expect(WhereToRun.flutterGemmaUrl, 'https://flutteredge.ai');
  });

  group('/llms.txt', () {
    final text = llmsTxt(version: '1.2.3');

    test('opens with one H1 and a one-line summary', () {
      final lines = text.split('\n');
      expect(lines[0], '# litetune');
      expect(lines[1], '');
      expect(lines[2], '> $kLandingDescription');
      // Outside the fenced blocks, where `#` is a shell comment.
      var fenced = false;
      final headings = <String>[];
      for (final line in lines) {
        if (line.startsWith('```')) {
          fenced = !fenced;
        } else if (!fenced && line.startsWith('# ')) {
          headings.add(line);
        }
      }
      expect(headings, ['# litetune']);
    });

    test('lists every measured model and the version', () {
      expect(text, contains('version 1.2.3'));
      for (final run in [...LlmModels.measured, ...EmbeddingModels.measured]) {
        expect(text, contains('`${run.hubId}`'));
      }
    });

    test('links only pages the build renders and known addresses', () {
      final links = RegExp(
        r'\]\(([^)]+)\)',
      ).allMatches(text).map((m) => m.group(1)!).toList();
      expect(links, isNotEmpty);
      for (final link in links) {
        if (link.startsWith(kSiteOrigin)) {
          final path = link.substring(kSiteOrigin.length);
          expect(_routes, contains(path), reason: '$link is not a route');
        } else {
          expect(
            [kReadmeUrl, kRepositoryUrl, kPypiUrl],
            contains(link),
            reason: '$link is not a known address',
          );
        }
      }
      for (final page in [
        '/measurements',
        ModelsTab.llm.path,
        '/changelog',
        kLlmsFullTxtPath,
      ]) {
        expect(links, contains('$kSiteOrigin$page'));
      }
      expect(links, contains(kReadmeUrl));
    });

    test('is served as plain text and kept out of the sitemap', () async {
      final response = await renderComponent(
        const TextFile('plain'),
        standalone: true,
      );
      expect(utf8.decode(response.body), 'plain');
      expect(response.headers['jaspr-sitemap-data'], ['false']);
      expect(response.headers['Content-Type'], ['text/plain; charset=utf-8']);
    });

    test('/llms-full.txt is the README and the measurements', () {
      final full = llmsFullTxt(readme: readme, measurements: measurements);
      expect(full, startsWith(readme));
      expect(full, endsWith(measurements));
    });
  });
}
