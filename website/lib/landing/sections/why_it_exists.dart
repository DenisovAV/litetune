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
  String what,
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
/// Each card opens a panel. It names the checkpoint the run actually used --
/// the Hub id and, where `MEASUREMENTS.md` pins one, the revision -- and links
/// to the model on the Hub and to the section that measured it. It
/// carries no score either, for the reason the face of the card carries none:
/// a panel one click deep is still the shortest place there is to quote a
/// verdict, and a reader who wants the number is one link from the table with
/// its interval, its sample size and its refusals next to it.
///
/// The panel opens over the page rather than under the card, and it does it
/// with `:target` -- the card is a link to the panel's own id, and CSS shows
/// that panel while the URL names it. No `@client` component, no island, no
/// script: the site builds in `static` mode and this page is not the one that
/// ends that. `<dialog>` would have been the semantically right element and
/// needs `showModal()` to behave as one, which is the trade this makes.
///
/// What that costs, said plainly. There is no focus trap and no
/// Escape-to-close, because both are JavaScript; closing is a link, of which
/// there are two -- the scrim behind the panel and the × in its corner -- and
/// both are reachable from the keyboard. Opening a panel puts a fragment in
/// the URL, so a reader who opened one and shared the address shares it open.
/// On paper nothing is open, as before.
///
/// `role="dialog"` labels the element. `aria-modal="true"` does not describe
/// the styling -- it tells assistive tech to treat everything outside as
/// inert, which nothing here makes true; it is set as the same trade as the
/// missing focus trap. `tabindex="-1"` is what makes the panel a focusable
/// area, so navigating to the fragment lands focus on it rather than leaving
/// it on the card behind.
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
        div(
          classes: 'measured',
          attributes: const {'id': closeTarget},
          [
            div(classes: 'measured-label', [
              Component.text('Measured end to end so far'),
            ]),
            div(classes: 'measured-cards', [
              for (final model in measured) _card(model),
            ]),
            p(classes: 'measured-note', [
              Component.text(
                'Measured on CPU, the conversion cost came out small at eight bits '
                'on all six, and at these sample sizes it often does not resolve '
                'at all. Four bits cost more, and how much more does not follow '
                'from the family or the parameter count. ',
              ),
              a(
                href:
                    'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md',
                attributes: _newTab,
                [Component.text('The numbers, and what they do not establish')],
              ),
            ]),
            for (final model in measured) _modal(model),
          ],
        ),
      ]),
    ]);
  }

  /// The runs, in the order the cards show them.
  static const measured = <MeasuredRun>[
    (
      name: 'FunctionGemma 270M',
      how: 'tool-call scoring, 640 held-out rows',
      what:
          '270M, text — built for tool calls: picking a function and filling its '
          'arguments on the device.',
      hubId: 'google/functiongemma-270m-it',
      revision: null,
      anchor: 'the-headline-numbers',
    ),
    (
      name: 'Gemma 3 270M',
      how: 'exact-text scoring, 600 held-out rows',
      what:
          '270M, text — the smallest here that still learns a task. Classification '
          'and routing, not conversation.',
      hubId: 'google/gemma-3-270m-it',
      revision: 'ac82b4e8',
      anchor: 'a-second-family-and-the-second-scorer',
    ),
    (
      name: 'Gemma 3 1B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '1B, text — the same export rule as the 270M, for when the 270M stops '
          'holding the task.',
      hubId: 'google/gemma-3-1b-it',
      revision: 'dcc83ea8',
      anchor: 'the-same-family-four-times-the-size',
    ),
    (
      name: 'Qwen3 0.6B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what: '0.6B, text — a first try when the task is labels.',
      hubId: 'Qwen/Qwen3-0.6B',
      revision: 'c1899de2',
      anchor: 'a-fourth-family-and-the-first-that-is-not-gemma',
    ),
    (
      name: 'Qwen2.5 0.5B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '0.5B, text — reach for it when memory is the budget: its four-bit export '
          'is the only one here that scored.',
      hubId: 'Qwen/Qwen2.5-0.5B-Instruct',
      revision: '7ae55760',
      anchor:
          'a-fifth-family-and-the-first-where-channelwise-four-bits-answered',
    ),
    (
      name: 'Gemma 4 E2B',
      how: 'exact-text scoring, the same 600 held-out rows',
      what:
          '5B, multimodal — text, vision and audio. For an assistant that has to '
          'see or hear.',
      hubId: 'google/gemma-4-E2B-it',
      // Null although MEASUREMENTS.md names a commit: that run passed no
      // `--revision` and the hash was read back from the cache afterwards.
      // Showing it here in the same slot as four pinned ones would give it a
      // parity the section it links to explicitly denies.
      revision: null,
      anchor: 'end-to-end-on-the-seven-projection-checkpoint',
    ),
  ];

  static const _newTab = {'target': '_blank', 'rel': 'noopener'};

  /// Where both close links point, and the id of the block they land on.
  /// A literal in three places was three chances for a panel that cannot be
  /// closed: `:target` stops matching when the fragment names nothing, so a
  /// renamed container leaves the × and the scrim pointing at dead air while
  /// every test still passes.
  static const closeTarget = 'measured';

  /// The panel's id, and what the card links to. Taken from the Hub id rather
  /// than the display name: the six last segments are already unique and are
  /// the thing that identifies the run, while two cards could one day differ
  /// in name alone. The `.` two Qwen ids carry is legal in an id and in a
  /// fragment, and is inert only because nothing selects these by id -- the
  /// stylesheet matches on class.
  static String slugFor(MeasuredRun model) =>
      'measured-${model.hubId.split('/').last.toLowerCase()}';

  static Component _card(MeasuredRun model) =>
      a(classes: 'measured-card', href: '#${slugFor(model)}', [
        span(classes: 'measured-card-model', [Component.text(model.name)]),
        span(classes: 'measured-card-what', [Component.text(model.what)]),
        span(classes: 'measured-card-how', [Component.text(model.how)]),
        span(classes: 'measured-card-more', [Component.text('checkpoint')]),
      ]);

  static Component _modal(MeasuredRun model) => div(
    classes: 'measured-modal',
    attributes: {
      'id': slugFor(model),
      'role': 'dialog',
      'aria-modal': 'true',
      // Without this the div is not a focusable area, so the browser's
      // navigate-to-fragment focusing step falls through to the viewport and
      // a screen reader is told a dialog opened while the cursor stays outside
      // it.
      'tabindex': '-1',
      'aria-label': '${model.name}: the checkpoint this run used',
    },
    [
      a(
        classes: 'measured-modal-scrim',
        href: '#$closeTarget',
        attributes: const {'aria-label': 'Close'},
        const [],
      ),
      div(classes: 'measured-modal-box', [
        div(classes: 'measured-modal-head', [
          span(classes: 'measured-modal-title', [Component.text(model.name)]),
          a(
            classes: 'measured-modal-close',
            href: '#$closeTarget',
            attributes: const {'aria-label': 'Close'},
            [Component.text('\u00d7')],
          ),
        ]),
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
          a(
            href: 'https://huggingface.co/${model.hubId}',
            attributes: _newTab,
            [Component.text('On Hugging Face')],
          ),
          a(
            href:
                'https://github.com/DenisovAV/litetune/blob/main/MEASUREMENTS.md#${model.anchor}',
            attributes: _newTab,
            [Component.text('What this run established')],
          ),
        ]),
      ]),
    ],
  );

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
      raw: const {
        'grid-template-columns': 'repeat(auto-fit, minmax(16rem, 1fr))',
      },
    ),
    css('.measured-card').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.3.rem),
      padding: Padding.all(1.rem),
      backgroundColor: Brand.surface,
      radius: BorderRadius.circular(0.6.rem),
      border: Border.all(color: Brand.line, width: 1.px),
      cursor: Cursor.pointer,
      raw: const {'text-decoration': 'none'},
    ),
    // The affordance, as a line of the card rather than a generated
    // `::after`: the old one hung off `summary` and its text was read out on
    // top of the platform's own expanded/collapsed announcement. A link needs
    // no such announcement, so the word can simply be in the card.
    css('.measured-card-more').styles(color: Brand.muted, fontSize: 0.85.rem),
    // Hidden until the URL names it. `:target` is the whole mechanism -- see
    // the class docstring for what that buys and what it costs.
    css('.measured-modal').styles(raw: const {'display': 'none'}),
    css('.measured-modal:target').styles(
      raw: const {
        'display': 'flex',
        'position': 'fixed',
        'inset': '0',
        'z-index': '50',
        'align-items': 'center',
        'justify-content': 'center',
        'padding': '1.5rem',
      },
    ),
    // The scrim is a link, so clicking beside the panel closes it. It is
    // drawn behind the box by source order, which is why the box is `relative`
    // rather than the scrim being negatively stacked.
    css('.measured-modal-scrim').styles(
      raw: const {
        'position': 'absolute',
        'inset': '0',
        'background': 'rgba(0, 0, 0, 0.45)',
      },
    ),
    css('.measured-modal-box').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.6.rem),
      padding: Padding.all(1.25.rem),
      backgroundColor: Brand.surface,
      radius: BorderRadius.circular(0.7.rem),
      border: Border.all(color: Brand.line, width: 1.px),
      raw: const {
        'position': 'relative',
        'width': 'min(30rem, 100%)',
        'max-height': '80vh',
        'overflow-y': 'auto',
      },
    ),
    css('.measured-modal-head').styles(
      display: Display.flex,
      alignItems: AlignItems.center,
      gap: Gap.all(1.rem),
      raw: const {'justify-content': 'space-between'},
    ),
    css(
      '.measured-modal-title',
    ).styles(color: Brand.ink, fontSize: 1.05.rem, fontWeight: FontWeight.w500),
    css('.measured-modal-close').styles(
      color: Brand.muted,
      fontSize: 1.3.rem,
      lineHeight: 1.em,
      raw: const {'text-decoration': 'none'},
    ),
    css(
      '.measured-card-model',
    ).styles(color: Brand.ink, fontSize: 1.05.rem, fontWeight: FontWeight.w500),
    css('.measured-card-what').styles(
      color: Brand.body,
      fontSize: 0.9.rem,
      raw: const {'line-height': '1.45'},
    ),
    // Demoted a step now that the line above it carries the reading: how a run
    // was scored is provenance, and it stops competing with what the model is.
    css('.measured-card-how').styles(color: Brand.muted, fontSize: 0.8.rem),
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
