import 'package:jaspr/dom.dart';

/// Design tokens for the litetune website.
///
/// The palette is monochrome on purpose: litetune's whole claim is that a
/// converted model can look fine and be worse, so the page reports rather than
/// sells. With no accent hue available, emphasis is carried by weight, size and
/// rules — see the sections, none of which introduce a colour of their own.
///
/// Every value is a CSS custom property rather than a literal, because the page
/// ships in two themes and a literal would have to be written twice. The
/// variables are declared once in [rootStyles]; components reference them
/// through the [Color] getters below and never spell a hex.
abstract final class Brand {
  // Colours, as references to the custom properties declared in [rootStyles].
  static const bg = Color('var(--lt-bg)');
  static const surface = Color('var(--lt-surface)');
  static const ink = Color('var(--lt-ink)');
  static const body = Color('var(--lt-body)');
  static const muted = Color('var(--lt-muted)');
  static const line = Color('var(--lt-line)');

  /// Body and headings. Related to the wordmark's face without being it: the
  /// logotype is Plex *Mono*, the page is Plex *Sans* — same superfamily, so
  /// the kinship reads, but the headline never impersonates the logo.
  static const fontSans = FontFamily.list([
    FontFamily('IBM Plex Sans'),
    FontFamilies.systemUi,
    FontFamilies.sansSerif,
  ]);

  /// The wordmark, command names, file extensions, labels and anything a
  /// reader might retype.
  static const fontMono = FontFamily.list([
    FontFamily('IBM Plex Mono'),
    FontFamily('SF Mono'),
    FontFamilies.monospace,
  ]);

  /// One theme, dark, unconditionally.
  ///
  /// An earlier version followed `prefers-color-scheme`, which is the usual
  /// advice and was wrong here: a visitor on a light desktop got a light page,
  /// and the site has one look on purpose. `color-scheme: dark` still tells the
  /// browser to paint form controls and scrollbars to match.
  static List<StyleRule> get rootStyles => [
    css(':root').styles(
      raw: const {
        '--lt-bg': '#0B0B0A',
        '--lt-surface': '#141413',
        '--lt-ink': '#F5F5F3',
        '--lt-body': '#C9C6BF',
        '--lt-muted': '#9A968E',
        '--lt-line': '#2A2A27',
        'color-scheme': 'dark',
      },
    ),
  ];
}
