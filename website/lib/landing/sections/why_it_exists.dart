import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../models/embedding_models.dart';
import '../../models/llm_models.dart';
import '../../models/models_page.dart';
import '../../theme/brand.dart';

/// The one paragraph that explains the tool's reason to exist, and a pointer
/// to the runs where it was actually done.
///
/// The paragraph carries no numbers, deliberately: the measured figures live
/// in MEASUREMENTS.md, where a reader can see the intervals and the sample
/// sizes next to them, and quoting a single score here would be the kind of
/// decontextualised claim the tool exists to argue against.
///
/// The model cards that used to follow it are on `/models` now (see
/// `lib/models/`). One card per family made this page grow with every
/// release; what stays is one line whose counts come from the lists the
/// models page renders, so a new card changes the count here without an edit
/// to this file.
class WhyItExists extends StatelessComponent {
  const WhyItExists({super.key});

  /// The line under the paragraph, counted from the cards on `/models`.
  static String get teaser {
    final llms = LlmModels.measured.length;
    final embeddings = EmbeddingModels.measured.length;
    return 'Measured end to end so far: $llms language '
        '${llms == 1 ? 'model' : 'models'} and $embeddings embedding '
        '${embeddings == 1 ? 'model' : 'models'}, each converted and compared '
        'with its float reference on held-out data.';
  }

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      div(classes: 'label', [Component.text('Why it exists')]),
      div(classes: 'why-body', [
        p(classes: 'prose', [
          Component.text(
            'Converting a model so it fits on a device changes it. The file still '
            'loads, the tools still report success, and the model can still be '
            'worse at your task than it was before. litetune runs the converted '
            'model on data it was never trained on, compares it with the version '
            'it came from, and tells you the difference.',
          ),
        ]),
        div(classes: 'why-models', [
          p(classes: 'why-models-line', [Component.text(teaser)]),
          div(classes: 'why-models-links', [
            a(href: ModelsTab.llm.path, [Component.text('Language models')]),
            a(href: ModelsTab.embeddings.path, [
              Component.text('Embedding models'),
            ]),
            a(href: '/measurements', [Component.text('The numbers')]),
          ]),
        ]),
      ]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.why-body').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(2.rem),
      raw: const {'max-width': '62ch'},
    ),
    css('.prose').styles(
      fontSize: 1.125.rem,
      lineHeight: 1.7.em,
      color: Brand.body,
      raw: const {'max-width': '62ch'},
      margin: Margin.zero,
    ),
    css('.why-models').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.75.rem),
      padding: Padding.all(1.rem),
      backgroundColor: Brand.surface,
      radius: BorderRadius.circular(0.6.rem),
      border: Border.all(color: Brand.line, width: 1.px),
    ),
    css('.why-models-line').styles(
      color: Brand.body,
      fontSize: 0.95.rem,
      margin: Margin.zero,
      raw: const {'line-height': '1.55'},
    ),
    css('.why-models-links').styles(
      display: Display.flex,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(1.25.rem),
      fontFamily: Brand.fontMono,
      fontSize: 0.85.rem,
    ),
    css(
      '.why-models-links a',
    ).styles(color: Brand.ink, raw: const {'text-underline-offset': '3px'}),
  ];
}
