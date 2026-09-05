import 'dart:convert';

import 'package:jaspr/dom.dart' show RawText;
import 'package:jaspr/jaspr.dart';

/// Canonical production origin. Every SEO tag points here rather than at a
/// hosting-provider URL, so indexing settles on the custom domain.
const String kSiteOrigin = 'https://litetune.dev';

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

/// Builds the `<head>` tags for the landing page: canonical, robots,
/// Open Graph, Twitter Card and JSON-LD.
List<Component> seoHead({
  required String title,
  required String description,
  String path = '/',
  String image = '/images/og-image.png',
}) {
  final url = '$kSiteOrigin$path';
  final imageUrl = '$kSiteOrigin$image';

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
    meta('twitter:card', 'summary_large_image'),
    meta('twitter:title', title),
    meta('twitter:description', description),
    meta('twitter:image', imageUrl),
    Component.element(
      tag: 'script',
      attributes: const {'type': 'application/ld+json'},
      children: [
        RawText(
          jsonEncode({
            '@context': 'https://schema.org',
            '@type': 'SoftwareApplication',
            'name': 'litetune',
            'applicationCategory': 'DeveloperApplication',
            'operatingSystem': 'macOS, Linux',
            'description': description,
            'url': kSiteOrigin,
            'license': 'https://www.apache.org/licenses/LICENSE-2.0',
            'author': {'@type': 'Person', 'name': 'Sasha Denisov'},
            'offers': {'@type': 'Offer', 'price': '0', 'priceCurrency': 'USD'},
          }),
        ),
      ],
    ),
  ];
}
