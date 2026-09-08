import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The one paragraph that explains the tool's reason to exist, and the two
/// runs where it was actually done.
///
/// The paragraph carries no numbers, deliberately: the measured figures live
/// in MEASUREMENTS.md, where a reader can see the intervals and the sample
/// sizes next to them, and quoting a single score here would be the kind of
/// decontextualised claim the tool exists to argue against.
///
/// The cards below it hold to the same rule. The only quantity on them is a
/// sample size, which is the thing a claim needs rather than a claim itself,
/// and neither card carries a score or a verdict. That is not squeamishness:
/// `MEASUREMENTS.md` records the FunctionGemma conversion cost resolving in
/// 2 of 6 recipe-runs across three runs of the same configuration, calls
/// whether 640 examples resolve it "close to a coin flip", and says in terms
/// that one run's verdict should not be quoted as the answer. A card is the
/// shortest place there is to quote a verdict, so it quotes none.
///
/// What each card carries instead is the fact that is unambiguous and belongs
/// to that run alone. FunctionGemma was also run on a phone, on the same 640
/// rows, which Gemma 3 270M has no counterpart to. Gemma 3 270M's untuned base
/// could not be scored at all -- it repeated itself on 571 of 600 prompts and
/// `verify` stopped before the quality tier -- which is why its training gain
/// is unattributable in principle rather than merely unmeasured.
///
/// The card says "verify refused" and not "the base could not be scored". The
/// refusal is the tool doing its job, and the passive form hands the failure
/// to the model: an exact-match score against those generations would have
/// been a number near zero, and it would have read as "bad at this task"
/// rather than "never answered in the shape the task requires".
/// `MEASUREMENTS.md` uses the active verb for the same reason.
///
/// The size is in the card name because `models.py` scopes the `gemma-3-text`
/// family to the 270M and the 1B, and only the 270M was measured.
class WhyItExists extends StatelessComponent {
  const WhyItExists({super.key});

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
        div(classes: 'measured', [
          div(classes: 'measured-label', [
            Component.text('Measured end to end so far'),
          ]),
          div(classes: 'cards', [
            _card(
              'FunctionGemma 270M',
              'tool-call scoring, 640 held-out rows',
              'Also run on a Snapdragon Galaxy S24.',
            ),
            _card(
              'Gemma 3 270M',
              'exact-text scoring, 600 held-out rows',
              'verify refused to score the untuned base at all.',
            ),
          ]),
          p(classes: 'measured-note', [
            Component.text(
              'Both on CPU. Conversion cost came out small on both, and at '
              'these sample sizes the method is near its limit. ',
            ),
            a(
              href:
                  'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md',
              attributes: const {'target': '_blank', 'rel': 'noopener'},
              [Component.text('The numbers, and what they do not establish')],
            ),
          ]),
        ]),
      ]),
    ]);
  }

  static Component _card(String model, String how, String note) =>
      div(classes: 'card', [
        div(classes: 'card-model', [Component.text(model)]),
        div(classes: 'card-how', [Component.text(how)]),
        div(classes: 'card-note', [Component.text(note)]),
      ]);

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
    css('.measured-label').styles(
      color: Brand.muted,
      fontSize: 0.9.rem,
      margin: Margin.only(bottom: 0.75.rem),
    ),
    css('.cards').styles(
      display: Display.grid,
      gap: Gap.all(0.75.rem),
      raw: const {
        'grid-template-columns': 'repeat(auto-fit, minmax(16rem, 1fr))',
      },
    ),
    css('.card').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.3.rem),
      padding: Padding.all(1.rem),
      backgroundColor: Brand.surface,
      radius: BorderRadius.circular(0.6.rem),
      border: Border.all(color: Brand.line, width: 1.px),
    ),
    css(
      '.card-model',
    ).styles(color: Brand.ink, fontSize: 1.05.rem, fontWeight: FontWeight.w500),
    css('.card-how').styles(color: Brand.body, fontSize: 0.9.rem),
    css(
      '.card-note',
    ).styles(color: Brand.muted, fontSize: 0.9.rem, lineHeight: 1.45.em),
    css('.measured-note').styles(
      color: Brand.muted,
      fontSize: 0.9.rem,
      lineHeight: 1.55.em,
      margin: Margin.only(top: 0.75.rem, bottom: Unit.zero),
    ),
    css('.measured-note a').styles(color: Brand.body),
  ];
}
