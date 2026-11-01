import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../landing/sections/nav_bar.dart';
import '../landing/sections/site_footer.dart';
import '../theme/brand.dart';
import 'embedding_models.dart';
import 'llm_models.dart';

/// The two halves of `/models`, each a route of its own.
///
/// The site builds in `static` mode with no client components, so the switch
/// between them cannot be a tab that swaps content in the browser. It is two
/// links to two pre-rendered pages that share this layout, and the half a page
/// renders is the one its link is drawn active for. That also gives each half
/// its own address, title and description, which a script-driven tab would
/// not.
enum ModelsTab {
  llm(path: '/models', label: 'LLMs'),
  embeddings(path: '/models/embeddings', label: 'Embeddings');

  const ModelsTab({required this.path, required this.label});

  /// The route this half is rendered at, and what its switch link points to.
  final String path;

  /// The switch link's text.
  final String label;
}

/// `/models` and `/models/embeddings`: the models litetune has measured end to
/// end, language models and embedding models behind one switch.
class ModelsPage extends StatelessComponent {
  const ModelsPage({required this.version, required this.tab, super.key});

  /// The package version, shown in the nav bar.
  final String version;

  /// Which half this page renders, and which switch link is drawn active.
  final ModelsTab tab;

  static const intro =
      'Each model here was converted with litetune and compared with its '
      'float reference on held-out data. A card says what the model is and '
      'which checkpoint the run used, and links to the section that measured it.';

  @override
  Component build(BuildContext context) {
    return div(classes: 'page', [
      NavBar(version: version),
      main_(classes: 'content', [
        section(classes: 'models', [
          h1(classes: 'models-title', [Component.text('Models')]),
          p(classes: 'models-intro', [Component.text(intro)]),
          nav(
            classes: 'models-switch',
            attributes: const {'aria-label': 'Model type'},
            [for (final t in ModelsTab.values) _switchLink(t)],
          ),
          switch (tab) {
            ModelsTab.llm => const LlmModels(),
            ModelsTab.embeddings => const EmbeddingModels(),
          },
        ]),
      ]),
      div(classes: 'content', [const SiteFooter()]),
    ]);
  }

  Component _switchLink(ModelsTab t) => a(
    href: t.path,
    classes: t == tab
        ? 'models-switch-link models-switch-active'
        : 'models-switch-link',
    attributes: t == tab ? const {'aria-current': 'page'} : const {},
    [Component.text(t.label)],
  );

  @css
  static List<StyleRule> get styles => [
    css('.models').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.5.rem),
      padding: Padding.symmetric(vertical: 3.rem),
    ),
    css('.models-title').styles(
      fontSize: 1.9.rem,
      fontWeight: FontWeight.w600,
      color: Brand.ink,
      margin: Margin.zero,
      raw: const {'line-height': '1.2'},
    ),
    css('.models-intro').styles(
      color: Brand.body,
      fontSize: 1.05.rem,
      margin: Margin.zero,
      raw: const {'line-height': '1.65', 'max-width': '62ch'},
    ),
    // A segmented control drawn from two links: one outline, the active half
    // filled. Monochrome like the rest of the site, so the active state is
    // carried by fill and weight rather than by a hue.
    css('.models-switch').styles(
      display: Display.inlineFlex,
      padding: Padding.all(0.2.rem),
      border: Border.all(color: Brand.line, width: 1.px),
      fontFamily: Brand.fontMono,
      fontSize: 0.85.rem,
      margin: Margin.only(bottom: 0.5.rem),
      raw: const {'border-radius': '999px', 'align-self': 'flex-start'},
    ),
    css('.models-switch-link').styles(
      padding: Padding.symmetric(horizontal: 1.rem, vertical: 0.4.rem),
      color: Brand.muted,
      textDecoration: TextDecoration.none,
      raw: const {'border-radius': '999px'},
    ),
    css('.models-switch-link:hover').styles(color: Brand.ink),
    css('.models-switch-active').styles(
      backgroundColor: Brand.surface,
      color: Brand.ink,
      fontWeight: FontWeight.w500,
      raw: const {'box-shadow': 'inset 0 0 0 1px var(--lt-muted)'},
    ),
    css.media(MediaQuery.screen(maxWidth: 640.px), [
      css('.models').styles(padding: Padding.only(top: 1.5.rem)),
      css('.models-title').styles(fontSize: 1.5.rem),
    ]),
  ];
}
