import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../theme/brand.dart';

/// One end-to-end run, as the card and its panel say it.
///
/// `revision` is null where `MEASUREMENTS.md` pins none -- FunctionGemma's
/// runs record the dataset and the split but never a base-model commit -- and
/// the panel omits the line rather than showing an empty one or a revision
/// nobody wrote down. `anchor` is GitHub's own heading anchor for the section
/// that measured this model; `measured_cards_test.dart` checks that the
/// section still exists and still names this checkpoint.
///
/// `runtimes` is where the checkpoint has run, each with the anchor of a
/// section that names both the checkpoint and that runtime. Having run is all
/// an entry says: the NPU runs were not scored on the task the card's own run
/// was, and the panel's heading, "Where it has run", claims no more than that.
/// The panel lists them in [RunTarget] order whatever order they are written
/// in.
typedef MeasuredRun = ({
  String name,
  String what,
  String how,
  String hubId,
  String? revision,
  String anchor,
  List<RanOn> runtimes,
});

/// One runtime a checkpoint has run on, and the section that says so.
typedef RanOn = ({RunTarget target, String anchor});

/// The runtimes a panel can name, in the order it names them.
enum RunTarget {
  cpu('CPU'),
  macGpu('Mac GPU'),
  qualcommNpu('Qualcomm NPU (SM8750)');

  const RunTarget(this.label);

  /// The link's text in the panel.
  final String label;
}

/// The cards of measured runs, each opening a panel about its model.
///
/// One component for both halves of `/models`, so that a language model and
/// an embedding model are presented the same way and a fix to the panel
/// reaches both.
///
/// Each card opens a panel. It says what the model is and what it is for,
/// names the checkpoint the run actually used -- the Hub id and, where
/// `MEASUREMENTS.md` pins one, the revision -- lists where that checkpoint has
/// run, each runtime a link to the section that ran it, and links to the
/// model on the Hub and to the section that measured it. It
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
class MeasuredCards extends StatelessComponent {
  const MeasuredCards({
    required this.page,
    required this.label,
    required this.runs,
    this.note = const [],
    super.key,
  });

  /// The path of the page the cards are on. Every page carries
  /// `<base href="/">` (jaspr's `Document.base` default, which the CSP in
  /// firebase.json relies on), so a bare `#fragment` resolves against `/`
  /// and would leave the page; the fragments are prefixed with this.
  final String page;

  /// The line above the cards, saying what they are a list of.
  final String label;

  /// The runs, in the order the cards show them.
  final List<MeasuredRun> runs;

  /// What follows the cards: the scope of the runs behind them, and where the
  /// numbers are. Empty renders no paragraph.
  final List<Component> note;

  @override
  Component build(BuildContext context) {
    return div(
      classes: 'measured',
      attributes: const {'id': closeTarget},
      [
        div(classes: 'measured-label', [Component.text(label)]),
        div(classes: 'measured-cards', [for (final run in runs) _card(run)]),
        if (note.isNotEmpty) p(classes: 'measured-note', note),
        for (final run in runs) _modal(run),
      ],
    );
  }

  static const _newTab = {'target': '_blank', 'rel': 'noopener'};

  /// Where both close links point, and the id of the block they land on.
  /// A literal in three places was three chances for a panel that cannot be
  /// closed: `:target` stops matching when the fragment names nothing, so a
  /// renamed container leaves the × and the scrim pointing at dead air while
  /// every test still passes.
  static const closeTarget = 'measured';

  /// The panel's id, and what the card links to. Taken from the Hub id rather
  /// than the display name: the twelve last segments are already unique and are
  /// the thing that identifies the run, while two cards could one day differ
  /// in name alone. The `.` five ids carry is legal in an id and in a
  /// fragment, and is inert only because nothing selects these by id -- the
  /// stylesheet matches on class.
  static String slugFor(MeasuredRun model) =>
      'measured-${model.hubId.split('/').last.toLowerCase()}';

  Component _card(MeasuredRun model) =>
      a(classes: 'measured-card', href: '$page#${slugFor(model)}', [
        span(classes: 'measured-card-model', [Component.text(model.name)]),
        span(classes: 'measured-card-how', [Component.text(model.how)]),
        // What opening it gets you. It used to read `checkpoint`, which named
        // one row of the panel rather than the reason to open it.
        span(classes: 'measured-card-more', [
          Component.text('what it is, and what it is for'),
        ]),
      ]);

  /// [runs] in [RunTarget] order, so every panel lists CPU, then Mac GPU,
  /// then the NPU, however a card wrote them.
  static List<RanOn> _inOrder(List<RanOn> runs) =>
      [...runs]..sort((x, y) => x.target.index.compareTo(y.target.index));

  Component _modal(MeasuredRun model) => div(
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
      'aria-label':
          '${model.name}: what it is, and the checkpoint this run used',
    },
    [
      a(
        classes: 'measured-modal-scrim',
        href: '$page#$closeTarget',
        attributes: const {'aria-label': 'Close'},
        const [],
      ),
      div(classes: 'measured-modal-box', [
        div(classes: 'measured-modal-head', [
          span(classes: 'measured-modal-title', [Component.text(model.name)]),
          a(
            classes: 'measured-modal-close',
            href: '$page#$closeTarget',
            attributes: const {'aria-label': 'Close'},
            [Component.text('\u00d7')],
          ),
        ]),
        p(classes: 'measured-modal-what', [Component.text(model.what)]),
        div(classes: 'measured-card-fact', [
          span(classes: 'measured-card-key', [Component.text('Checkpoint')]),
          span(classes: 'measured-card-id', [Component.text(model.hubId)]),
        ]),
        if (model.revision case final revision?)
          div(classes: 'measured-card-fact', [
            span(classes: 'measured-card-key', [Component.text('Revision')]),
            span(classes: 'measured-card-id', [Component.text(revision)]),
          ]),
        if (model.runtimes.isNotEmpty)
          div(classes: 'measured-card-fact', [
            span(classes: 'measured-card-key', [
              Component.text('Where it has run'),
            ]),
            span(classes: 'measured-card-runs', [
              for (final (i, run) in _inOrder(model.runtimes).indexed) ...[
                if (i > 0) Component.text(' · '),
                a(href: '/measurements#${run.anchor}', [
                  Component.text(run.target.label),
                ]),
              ],
            ]),
          ]),
        div(classes: 'measured-card-links', [
          a(
            href: 'https://huggingface.co/${model.hubId}',
            attributes: _newTab,
            [Component.text('On Hugging Face')],
          ),
          a(href: '/measurements#${model.anchor}', [
            Component.text('Scores and intervals for this run'),
          ]),
        ]),
      ]),
    ],
  );

  // Every class here carries the `measured-` prefix, including the
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
    css('.measured-label').styles(
      color: Brand.muted,
      fontSize: 0.9.rem,
      margin: Margin.only(bottom: 0.75.rem),
    ),
    css('.measured-cards').styles(
      display: Display.grid,
      gap: Gap.all(0.75.rem),
      raw: const {
        // auto-fill keeps the empty tracks, so a list of one card is one column
        // wide instead of the page's width.
        'grid-template-columns': 'repeat(auto-fill, minmax(16rem, 1fr))',
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
    // The convention the rest of the site uses for a clickable card
    // (`where_to_run.dart`, `nav_bar.dart`, `site_footer.dart`): the border
    // brightens. Without it the only cue that a card is a link was a muted
    // caption -- which is the line this change is about.
    css('.measured-card:hover').styles(
      border: Border.all(color: Brand.muted, width: 1.px),
    ),
    css('.measured-card:hover .measured-card-more').styles(color: Brand.ink),
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
        'max-height': '80dvh',
        'overflow-y': 'auto',
      },
    ),
    // Sticky, because the box scrolls and the head carries the only explicit
    // way out. A one-line description fitted without scrolling; a paragraph
    // does not, and on a short viewport the visitor would have scrolled the
    // close control off the top of the thing they were trying to close. The
    // background is the box's own, so the text passes under it rather than
    // through it.
    css('.measured-modal-head').styles(
      display: Display.flex,
      alignItems: AlignItems.center,
      gap: Gap.all(1.rem),
      backgroundColor: Brand.surface,
      padding: Padding.only(bottom: 0.4.rem),
      raw: const {
        'justify-content': 'space-between',
        'position': 'sticky',
        'top': '0',
      },
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
    css('.measured-modal-what').styles(
      color: Brand.body,
      fontSize: 0.95.rem,
      // Both sides, not just the bottom. Nothing on this page resets `p`, so
      // the UA's `margin-block-start: 1em` applied -- and inside a flex column
      // margins do not collapse, so the paragraph sat further from the title
      // than from the row under it, which is backwards.
      margin: Margin.only(top: 0.px, bottom: 0.4.rem),
      raw: const {'line-height': '1.55'},
    ),
    css('.measured-card-how').styles(color: Brand.body, fontSize: 0.9.rem),
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
    css('.measured-card-runs').styles(color: Brand.muted, fontSize: 0.85.rem),
    css('.measured-card-runs a').styles(color: Brand.body),
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
