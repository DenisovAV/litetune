import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The one paragraph that explains the tool's reason to exist, and the runs
/// where it was actually done.
///
/// The paragraph carries no numbers, deliberately: the measured figures live
/// in MEASUREMENTS.md, where a reader can see the intervals and the sample
/// sizes next to them, and quoting a single score here would be the kind of
/// decontextualised claim the tool exists to argue against.
///
/// The cards below it hold to the same rule. The only quantity on them is a
/// sample size, which is the thing a claim needs rather than a claim itself,
/// and no card carries a score or a verdict. That is not squeamishness:
/// `MEASUREMENTS.md` records the FunctionGemma conversion cost resolving in
/// 2 of 6 recipe-runs across three runs of the same configuration, calls
/// whether 640 examples resolve it "close to a coin flip", and says in terms
/// that one run's verdict should not be quoted as the answer. A card is the
/// shortest place there is to quote a verdict, so it quotes none.
///
/// A card carries the model, the scorer and the sample size, and stops there.
/// It used to carry a third line apiece -- a phone run, a refused base, an
/// export that needed no flag -- and four cards' worth of those read as
/// footnotes rather than as one fact each, with nothing shared to compare
/// across. What is worth saying about a single run is worth a sentence in
/// `MEASUREMENTS.md`, which the link below the cards goes to.
///
/// The note under the cards says "at eight bits" because four bits did not
/// come out small: `MEASUREMENTS.md`'s "What four bits cost" has Gemma 3 270M
/// at +0.3483 and +0.3200 under the two block-wise recipes, both resolved.
/// Without the qualifier the note would say the opposite of that table. It
/// says "on CPU" because the same file records the Qwen3 bundles on a
/// phone as well, where the backend changes the answer: the GPU costs
/// +0.0483 on the 8-bit bundle where the phone's CPU costs +0.0167. And it
/// ends on the model rather than the family or the size because the same
/// four-bit recipes cost the 1B 8.83 points against the 270M's 34.83 -- one
/// Gemma 3 rule, two answers -- and because the four costs do not order by
/// parameter count either: 3.50 on a 0.6B Qwen3, 7.67 on a 0.5B Qwen2.5, 8.83
/// on the 1B. That is the whole reason a card carries no verdict.
///
/// The size is in the card name because `models.py` scopes the `gemma-3-text`
/// family to the 270M and the 1B, and both are now measured -- two cards that
/// differ in nothing but size. Qwen3's card carries its size because its rule
/// is scoped to the one size that was run, and Qwen2.5's for the same reason,
/// where the rule is scoped to the one size *and* variant.
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
            _card('FunctionGemma 270M', 'tool-call scoring, 640 held-out rows'),
            _card('Gemma 3 270M', 'exact-text scoring, 600 held-out rows'),
            _card(
              'Gemma 3 1B',
              'exact-text scoring, the same 600 held-out rows',
            ),
            _card(
              'Qwen3 0.6B',
              'exact-text scoring, the same 600 held-out rows',
            ),
            _card(
              'Qwen2.5 0.5B',
              'exact-text scoring, the same 600 held-out rows',
            ),
          ]),
          p(classes: 'measured-note', [
            Component.text(
              'Measured on CPU, the conversion cost came out small at eight bits '
              'on all five, and at these sample sizes the method is near its '
              'limit. Four bits cost more, and how much more depends on the '
              'model rather than on its family or its size. ',
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

  static Component _card(String model, String how) => div(classes: 'card', [
    div(classes: 'card-model', [Component.text(model)]),
    div(classes: 'card-how', [Component.text(how)]),
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
    css('.measured-note').styles(
      color: Brand.muted,
      fontSize: 0.9.rem,
      lineHeight: 1.55.em,
      margin: Margin.only(top: 0.75.rem, bottom: Unit.zero),
    ),
    css('.measured-note a').styles(color: Brand.body),
  ];
}
