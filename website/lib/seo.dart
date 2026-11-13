import 'dart:convert';

import 'package:jaspr/dom.dart' show RawText;
import 'package:jaspr/jaspr.dart';

import 'models/embedding_models.dart';
import 'models/llm_models.dart';
import 'models/measured_cards.dart';

/// Canonical production origin. Every SEO tag points here rather than at a
/// hosting-provider URL, so indexing settles on the custom domain.
const String kSiteOrigin = 'https://litetune.dev';

/// Platforms on which at least one litetune stage is supported. The landing
/// page separately names where the complete five-stage pipeline runs.
const String kOperatingSystems = 'macOS, Linux, Windows';

/// Where the source lives and where the package is published. The JSON-LD
/// names both, and `/llms.txt` links them.
const String kRepositoryUrl = 'https://github.com/DenisovAV/litetune';
const String kPypiUrl = 'https://pypi.org/project/litetune/';

/// The terms the package is filed under, in the JSON-LD. The same subjects as
/// `keywords` in pyproject.toml, which is what PyPI shows.
const List<String> kKeywords = [
  'fine-tuning',
  'LoRA',
  'LiteRT-LM',
  '.litertlm',
  'LiteRT',
  'Gemma',
  'FunctionGemma',
  'EmbeddingGemma',
  'Qwen',
  'quantization',
  'on-device',
  'edge AI',
  'NPU',
  'Android',
  'Flutter',
];

/// Each page's `<title>` and meta description. Public because `/llms.txt`
/// describes the pages with the same words.
///
/// The landing description is also the application's description in every
/// page's JSON-LD. It names the method both ways because `tune` takes both:
/// `--method full`, the default, and `--method lora` (`tune.METHODS`).
/// EmbeddingGemma 2 is outside the parenthesis because its tune is a full
/// fine-tune only.
const String kLandingTitle =
    'litetune — fine-tune Gemma and Qwen for on-device LiteRT-LM';

const String kLandingDescription =
    'litetune fine-tunes small models — Gemma 3, Gemma 4, Qwen and '
    'FunctionGemma (full or LoRA), and EmbeddingGemma 2 — converts them to '
    '.litertlm for on-device LiteRT-LM across quantization recipes, and '
    'compares the result with the float model on held-out data.';

const String kChangelogTitle = 'Changelog — litetune';

const String kChangelogDescription =
    'Every litetune release, newest first. A release that makes an earlier '
    'result wrong says so first, and says what to do about it.';

const String kMeasurementsTitle =
    'LiteRT-LM conversion measurements — litetune';

const String kMeasurementsDescription =
    'Published model-quality and runtime measurements of models converted to '
    '.litertlm, with intervals, sample sizes, refusals and the limits of what '
    'each run establishes.';

const String kModelsTitle = 'Language models measured on LiteRT-LM — litetune';

const String kEmbeddingsTitle =
    'Embedding models measured on LiteRT-LM — litetune';

/// The families on the cards, in card order: each card's name without its
/// size and variant -- `Gemma 3 270M` and `Gemma 3 1B` are `Gemma 3`.
List<String> _families(List<MeasuredRun> runs) {
  final size = RegExp(r'^(\d+(\.\d+)?[BM]|E\dB|Instruct)$');
  return {
    for (final run in runs)
      run.name.split(' ').where((word) => !size.hasMatch(word)).join(' '),
  }.toList();
}

String _list(List<String> items) => items.length < 2
    ? items.join()
    : '${items.sublist(0, items.length - 1).join(', ')} and ${items.last}';

/// Read from the cards, so a card added to `/models` is named here too.
final String kModelsDescription =
    'The language models litetune has converted to .litertlm and compared '
    'with their float references end to end — '
    '${_list(_families(LlmModels.measured))} — each with the checkpoint its '
    'run used, where it has run and a link to its measurements.';

final String kEmbeddingsDescription =
    'The embedding models litetune has fine-tuned, converted to .litertlm and '
    'compared with their float references end to end — '
    '${_list(_families(EmbeddingModels.measured))} — each with the checkpoint '
    'its run used, where it has run and a link to its measurements.';

/// The social card, `web/images/og-image.png`, drawn by `tool/og_image.py`.
const String kOgImagePath = '/images/og-image.png';
const int kOgImageWidth = 1200;
const int kOgImageHeight = 630;
const String kOgImageAlt =
    'litetune: fine-tune small models, convert to LiteRT-LM, verify against '
    'the float model on CPU and GPU.';

/// Google Fonts for the two Plex faces the page uses.
///
/// Preconnect first: the stylesheet at fonts.googleapis.com immediately pulls
/// font files from fonts.gstatic.com, and without the second hint that
/// connection is only opened after the CSS parses.
List<Component> fontLinks() => [
  Component.element(
    tag: 'link',
    attributes: const {
      'rel': 'preconnect',
      'href': 'https://fonts.googleapis.com',
    },
  ),
  Component.element(
    tag: 'link',
    attributes: const {
      'rel': 'preconnect',
      'href': 'https://fonts.gstatic.com',
      'crossorigin': '',
    },
  ),
  Component.element(
    tag: 'link',
    attributes: const {
      'rel': 'stylesheet',
      'href':
          'https://fonts.googleapis.com/css2'
          '?family=IBM+Plex+Mono:wght@300;400;500;600'
          '&family=IBM+Plex+Sans:wght@200;300;400;500'
          '&display=swap',
    },
  ),
];

/// Builds the `<head>` tags for a page: canonical, robots, Open Graph,
/// Twitter Card and JSON-LD.
///
/// [description] describes the page. The JSON-LD block describes the
/// application instead, so a page about something narrower -- the changelog --
/// passes [applicationDescription] and keeps the application's own there.
///
/// [version] is the package version the page was built from; the JSON-LD
/// gives it as `softwareVersion` when it is passed.
List<Component> seoHead({
  required String title,
  required String description,
  String? applicationDescription,
  String? version,
  String path = '/',
}) {
  final url = '$kSiteOrigin$path';
  final imageUrl = '$kSiteOrigin$kOgImagePath';

  Component meta(String key, String value, {bool property = false}) {
    return Component.element(
      tag: 'meta',
      attributes: {property ? 'property' : 'name': key, 'content': value},
    );
  }

  return [
    Component.element(
      tag: 'link',
      attributes: {'rel': 'canonical', 'href': url},
    ),
    Component.element(
      tag: 'link',
      attributes: const {
        'rel': 'icon',
        'type': 'image/svg+xml',
        'href': '/favicon.svg',
      },
    ),
    meta('robots', 'index, follow'),
    // Matches the dark default in brand.dart, so the browser chrome does not
    // flash white before the stylesheet lands.
    meta('theme-color', '#0B0B0A'),
    meta('og:type', 'website', property: true),
    meta('og:site_name', 'litetune', property: true),
    meta('og:locale', 'en_US', property: true),
    meta('og:title', title, property: true),
    meta('og:description', description, property: true),
    meta('og:url', url, property: true),
    meta('og:image', imageUrl, property: true),
    meta('og:image:type', 'image/png', property: true),
    meta('og:image:width', '$kOgImageWidth', property: true),
    meta('og:image:height', '$kOgImageHeight', property: true),
    meta('og:image:alt', kOgImageAlt, property: true),
    meta('twitter:card', 'summary_large_image'),
    meta('twitter:title', title),
    meta('twitter:description', description),
    meta('twitter:image', imageUrl),
    meta('twitter:image:alt', kOgImageAlt),
    Component.element(
      tag: 'script',
      attributes: const {'type': 'application/ld+json'},
      children: [
        RawText(
          jsonEncode(
            applicationJsonLd(
              description: applicationDescription ?? description,
              version: version,
            ),
          ),
        ),
      ],
    ),
  ];
}

/// The JSON-LD object every page carries: the application, not the page.
///
/// Typed as both `SoftwareApplication` and `SoftwareSourceCode`, because
/// `codeRepository` and `programmingLanguage` are properties of the second
/// and `softwareVersion`, `downloadUrl` and `operatingSystem` of the first.
Map<String, Object> applicationJsonLd({
  required String description,
  String? version,
}) => {
  '@context': 'https://schema.org',
  '@type': ['SoftwareApplication', 'SoftwareSourceCode'],
  'name': 'litetune',
  'applicationCategory': 'DeveloperApplication',
  'operatingSystem': kOperatingSystems,
  'description': description,
  'url': kSiteOrigin,
  'image': '$kSiteOrigin$kOgImagePath',
  if (version != null) 'softwareVersion': version,
  'codeRepository': kRepositoryUrl,
  'programmingLanguage': 'Python',
  'downloadUrl': kPypiUrl,
  'sameAs': [kRepositoryUrl, kPypiUrl],
  'keywords': kKeywords.join(', '),
  'license': 'https://www.apache.org/licenses/LICENSE-2.0',
  'author': {'@type': 'Person', 'name': 'Sasha Denisov'},
  'offers': {'@type': 'Offer', 'price': '0', 'priceCurrency': 'USD'},
};
