import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The one paragraph that explains the tool's reason to exist.
///
/// It carries no numbers. The measured figures live in MEASUREMENTS.md, where a
/// reader can see the intervals and the sample size next to them; quoting a
/// single number here would be the kind of decontextualised claim the tool
/// exists to argue against.
class WhyItExists extends StatelessComponent {
  const WhyItExists({super.key});

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      div(classes: 'label', [Component.text('Why it exists')]),
      p(classes: 'prose', [
        Component.text(
          'Converting a model so it fits on a device changes it. The file still '
          'loads, the tools still report success, and the model can still be '
          'worse at your task than it was before. litetune runs the converted '
          'model on data it was never trained on, compares it with the version '
          'it came from, and tells you the difference.',
        ),
      ]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.prose').styles(
      fontSize: 1.125.rem,
      lineHeight: 1.7.em,
      color: Brand.body,
      raw: const {'max-width': '62ch'},
      margin: Margin.zero,
    ),
  ];
}
