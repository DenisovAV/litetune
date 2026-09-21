import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// One end-to-end run, as the card and its panel say it.
///
/// `revision` is null where `MEASUREMENTS.md` pins none -- FunctionGemma's
/// runs record the dataset and the split but never a base-model commit -- and
/// the panel omits the line rather than showing an empty one or a revision
/// nobody wrote down. `anchor` is GitHub's own heading anchor for the section
/// that measured this model; `measured_cards_test.dart` checks that the
/// section still exists and still names this checkpoint.
typedef MeasuredRun = ({
  String name,
  String how,
  String hubId,
  String? revision,
  String anchor,
});

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
/// says what the cost does *not* follow from rather than what it does: the
/// same four-bit recipes cost the 1B 8.83 points against the 270M's 34.83 --
/// one Gemma 3 rule, two answers -- and the four costs do not order by
/// parameter count either: 3.50 on Qwen3-0.6B, 7.67 on Qwen2.5-0.5B, 8.83 on
/// the 1B, 34.83 on the 270M. What does determine it is not something these
/// runs separate, which is the whole reason a card carries no verdict.
///
/// The size is in the card name because `models.py` scopes the `gemma-3-text`
/// family to the 270M and the 1B, and both are now measured -- two cards that
/// differ in nothing but size. Qwen3's card carries its size because its rule
/// is scoped to the one size that was run, and Qwen2.5's for the same reason,
/// where the rule is scoped to the one size *and* variant.
///
/// Each card opens. The panel under it names the checkpoint the run actually
/// used -- the Hub id and, where `MEASUREMENTS.md` pins one, the revision --
/// and links to the model on the Hub and to the section that measured it. It
/// carries no score either, for the reason the face of the card carries none:
/// a panel one click deep is still the shortest place there is to quote a
/// verdict, and a reader who wants the number is one link from the table with
/// its interval, its sample size and its refusals next to it.
///
/// `<details>`, not a dialog and not an island. The site builds in `static`
/// mode with no `@client` component anywhere, so a disclosure that needs
/// JavaScript would make this the page that ends that -- for a panel the
/// browser already implements and operates from the keyboard. No card is
/// rendered `open`, which is the cost as well as the point: with scripting
/// off the panel still opens, but on paper it does not, and a printed page
/// carries the model names without the checkpoints. `.measured-cards` gets
/// `align-items: start` so an open card grows downward instead of stretching
/// the ones beside it.
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
          div(classes: 'measured-cards', [
            for (final model in measured) _card(model),
          ]),
          p(classes: 'measured-note', [
            Component.text(
              'Measured on CPU, the conversion cost came out small at eight bits '
              'on all five, and at these sample sizes the method is near its '
              'limit. Four bits cost more, and how much more does not follow '
              'from the family or the parameter count. ',
            ),
            a(
              href:
                  'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md',
              attributes: _newTab,
              [Component.text('The numbers, and what they do not establish')],
            ),
          ]),
        ]),
      ]),
    ]);
  }

  /// The runs, in the order the cards show them.
  static const measured = <MeasuredRun>[
    (
      name: 'FunctionGemma 270M',
      how: 'tool-call scoring, 640 held-out rows',
      hubId: 'google/functiongemma-270m-it',
      revision: null,
      anchor: 'the-headline-numbers',
    ),
    (
      name: 'Gemma 3 270M',
      how: 'exact-text scoring, 600 held-out rows',
      hubId: 'google/gemma-3-270m-it',
      revision: 'ac82b4e8',
      anchor: 'a-second-family-and-the-second-scorer',
    ),
    (
      name: 'Gemma 3 1B',
      how: 'exact-text scoring, the same 600 held-out rows',
      hubId: 'google/gemma-3-1b-it',
      revision: 'dcc83ea8',
      anchor: 'the-same-family-four-times-the-size',
    ),
    (
      name: 'Qwen3 0.6B',
      how: 'exact-text scoring, the same 600 held-out rows',
      hubId: 'Qwen/Qwen3-0.6B',
      revision: 'c1899de2',
      anchor: 'a-fourth-family-and-the-first-that-is-not-gemma',
    ),
    (
      name: 'Qwen2.5 0.5B',
      how: 'exact-text scoring, the same 600 held-out rows',
      hubId: 'Qwen/Qwen2.5-0.5B-Instruct',
      revision: '7ae55760',
      anchor:
          'a-fifth-family-and-the-first-where-channelwise-four-bits-answered',
    ),
  ];

  static const _newTab = {'target': '_blank', 'rel': 'noopener'};

  static Component _card(
    MeasuredRun model,
  ) => details(classes: 'measured-card', [
    summary(classes: 'measured-card-summary', [
      span(classes: 'measured-card-model', [Component.text(model.name)]),
      span(classes: 'measured-card-how', [Component.text(model.how)]),
    ]),
    div(classes: 'measured-card-panel', [
      div(classes: 'measured-card-fact', [
        span(classes: 'measured-card-key', [Component.text('Checkpoint')]),
        span(classes: 'measured-card-id', [Component.text(model.hubId)]),
      ]),
      if (model.revision case final revision?)
        div(classes: 'measured-card-fact', [
          span(classes: 'measured-card-key', [Component.text('Revision')]),
          span(classes: 'measured-card-id', [Component.text(revision)]),
        ]),
      div(classes: 'measured-card-links', [
        a(href: 'https://huggingface.co/${model.hubId}', attributes: _newTab, [
          Component.text('On Hugging Face'),
        ]),
        a(
          href:
              'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md#${model.anchor}',
          attributes: _newTab,
          [Component.text('What this run established')],
        ),
      ]),
    ]),
  ]);

  // Every class here carries the section's `measured-` prefix, including the
  // cards. jaspr collects each component's `@css` into one stylesheet, so a
  // class name is global: `where_to_run.dart` also styles `.cards` and
  // `.card`, and the two rules landed on the same elements -- whichever the
  // bundle emitted second won each property, which is why that section's
  // cards were drawn with this one's 1rem padding rather than their own
  // 1.4rem. Scoping the names here ends that for these elements, and the
  // `<details>` makes it matter more than it did: the other rule sets
  // `display: flex` and `flex: 1 1 240px`, which would otherwise apply to a
  // disclosure that wants neither.
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
    css('.measured-cards').styles(
      display: Display.grid,
      gap: Gap.all(0.75.rem),
      // An open card grows downward; the ones beside it keep their height.
      alignItems: AlignItems.start,
      raw: const {
        'grid-template-columns': 'repeat(auto-fit, minmax(16rem, 1fr))',
      },
    ),
    css('.measured-card').styles(
      padding: Padding.all(1.rem),
      backgroundColor: Brand.surface,
      radius: BorderRadius.circular(0.6.rem),
      border: Border.all(color: Brand.line, width: 1.px),
    ),
    // The default triangle is replaced by a sign that lines up with the
    // model name. Both the `display: flex` here and `list-style: none` below
    // can suppress a marker drawn as a list marker, and the WebKit
    // pseudo-element needs its own rule; which one is load-bearing depends on
    // the engine, and nothing in this repository tests that, so all three
    // stay.
    //
    // The accessible-name computation includes `::before` and `::after`
    // content, so the words are read out on top of the expanded/collapsed
    // state the platform already announces. Kept because they are what tell a
    // sighted reader there is anything to open, and recorded because nothing
    // here gives the control an explicit label to exclude them.
    css('.measured-card-summary').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.3.rem),
      cursor: Cursor.pointer,
      listStyle: ListStyle.none,
    ),
    css(
      '.measured-card-summary::-webkit-details-marker',
    ).styles(raw: const {'display': 'none'}),
    css(
      '.measured-card-summary::after',
    ).styles(color: Brand.muted, fontSize: 0.85.rem, content: '+ checkpoint'),
    css(
      '.measured-card[open] .measured-card-summary::after',
    ).styles(content: '− checkpoint'),
    css(
      '.measured-card-model',
    ).styles(color: Brand.ink, fontSize: 1.05.rem, fontWeight: FontWeight.w500),
    css('.measured-card-how').styles(color: Brand.body, fontSize: 0.9.rem),
    css('.measured-card-panel').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.4.rem),
      margin: Margin.only(top: 0.75.rem),
      padding: Padding.only(top: 0.75.rem),
      border: Border.only(
        top: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.measured-card-fact').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.1.rem),
    ),
    css('.measured-card-key').styles(color: Brand.muted, fontSize: 0.75.rem),
    css('.measured-card-id').styles(
      color: Brand.body,
      fontFamily: Brand.fontMono,
      fontSize: 0.85.rem,
      raw: const {'overflow-wrap': 'anywhere'},
    ),
    css('.measured-card-links').styles(
      display: Display.flex,
      flexWrap: FlexWrap.wrap,
      gap: Gap.all(0.9.rem),
      margin: Margin.only(top: 0.25.rem),
      fontSize: 0.85.rem,
    ),
    css('.measured-card-links a').styles(color: Brand.body),
    css('.measured-note').styles(
      color: Brand.muted,
      fontSize: 0.9.rem,
      lineHeight: 1.55.em,
      margin: Margin.only(top: 0.75.rem, bottom: Unit.zero),
    ),
    css('.measured-note a').styles(color: Brand.body),
  ];
}
