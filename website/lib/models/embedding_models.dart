import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../theme/brand.dart';
import 'measured_cards.dart';

/// One table of measured numbers, with the conditions it was taken under.
///
/// The conditions are a field of their own, not a caption a later edit can
/// drop: a number without them is the decontextualised claim this site exists
/// to argue against.
typedef MeasuredTable = ({
  String title,
  String conditions,
  List<String> header,
  List<List<String>> rows,
  List<String> notes,
});

/// What litetune does with one command, as the page lists it.
typedef EmbeddingStep = ({String command, String text});

/// `/models/embeddings`: EmbeddingGemma 2, the one embedding model litetune
/// fine-tunes, converts and verifies.
///
/// Every sentence and every number here is taken from two files and nothing
/// else: README.md (*Decisions on EmbeddingGemma 2* and *On a Qualcomm NPU:
/// `convert --npu`*) and MEASUREMENTS.md (*EmbeddingGemma 2: a decider
/// fine-tuned and grafted into Google's file*, *EmbeddingGemma 2 on a Qualcomm
/// NPU*, and the EmbeddingGemma 2 rows of *The CPU and the GPU on a Mac*).
/// `models_page_test.dart` checks that every numeral on this page occurs in one
/// of those two files, so a number typed here from memory fails the build's
/// tests rather than shipping.
///
/// Unlike the language-model cards, this page carries scores. They are not on
/// the card -- the card keeps the rule the others keep -- but in tables below
/// it, each under the conditions it was taken in: the task and its split, the
/// decider, the runtime and where it ran. That is the context MEASUREMENTS.md
/// puts beside them, and the reason the language-model cards send a reader
/// there instead: twelve families' worth of it does not fit one page, one
/// model's does.
///
/// The hardware is named only as far as README.md names it -- an x86 host's
/// CPU, a Mac's GPU, and the phone the NPU file ran on, which is the measured
/// device and not an environment.
class EmbeddingModels extends StatelessComponent {
  const EmbeddingModels({super.key});

  /// EmbeddingGemma 2 as a card, in the shape the language models use.
  ///
  /// The revision is the checkpoint's, as MEASUREMENTS.md pins it in *The
  /// graft* (`google/embeddinggemma-2` at `914f7f89`). The anchor is the
  /// section that introduces the path and holds the end-to-end runs.
  static const MeasuredRun embeddingGemma2 = (
    name: 'EmbeddingGemma 2',
    how: 'centroid decider, 2,617 held-out rows',
    what:
        "270M parameters: the text tower of Google's EmbeddingGemma 2. An app "
        "uses it through flutter_edge_ai's EmbeddingModel.createDecider, which "
        'embeds the input and answers with the option whose examples have the '
        'nearest centroid. litetune fine-tunes the tower on one decision and '
        "writes the result into Google's own .litertlm file, so the app loads "
        'one file and nothing ships beside it.',
    hubId: 'google/embeddinggemma-2',
    revision: '914f7f89',
    anchor:
        'embeddinggemma-2-a-decider-fine-tuned-and-grafted-into-googles-file',
  );

  /// The cards, in order. One today.
  static const measured = <MeasuredRun>[embeddingGemma2];

  static const summary =
      'Every score below is the accuracy of the centroid decider verify fits, '
      'on held-out rows, against the float32 fine-tune as the reference. ';

  /// README.md, *Decisions on EmbeddingGemma 2* and *On a Qualcomm NPU*.
  static const steps = <EmbeddingStep>[
    (
      command: 'tune',
      text:
          'Fine-tunes the whole text tower on a decision\'s rows, in float32, '
          'with the one recipe that was measured: there are no training flags. '
          'It runs in an environment of its own.',
    ),
    (
      command: 'convert',
      text:
          'Maps every weight and norm of the fine-tune onto Google\'s graph, '
          'writes them as int4 (the default, 164,626,432 bytes, Google\'s size) '
          'or int8 (300,089,344 bytes), and reads the result back against '
          'Google\'s file byte for byte outside what it wrote. A weight it '
          'cannot place is refused rather than left as Google\'s.',
    ),
    (
      command: 'verify',
      text:
          'Embeds every training and held-out row with the float32 fine-tune '
          'and with the file, on LiteRT-LM\'s CPU or, with --backend gpu, on a '
          'Mac\'s GPU. It fits the centroid decider on ten examples per option, '
          'five draws, and passes when every draw\'s upper 95 % bound on the '
          'accuracy difference is within 0.01.',
    ),
    (
      command: 'convert --npu',
      text:
          'Compiles the int8 file\'s text encoder ahead of time for a Qualcomm '
          'SoC: SM8550, SM8650, SM8750 or SM8850, the four Google publishes NPU '
          'files for. int8 only, on Linux x86-64 only. verify cannot run such a '
          'file; the SM8750 one has run on a phone, and no other SoC\'s file '
          'has run on a device.',
    ),
  ];

  /// MEASUREMENTS.md, in the order the sections run.
  static const tables = <MeasuredTable>[
    (
      title: 'banking77, through litetune\'s own stages',
      conditions:
          '77 options. prepare → tune → convert (int8 and int4) → verify '
          '--with-base, on prepare\'s split: 9,158 training and 2,617 held-out '
          'rows. Accuracy is the 10-shot centroid decider\'s mean over five '
          'draws; LiteRT-LM 0.18.0\'s EmbeddingEngine on an x86 host\'s CPU.',
      header: ['embeddings', '10-shot accuracy', 'conversion cost', 'verify'],
      rows: [
        ['Google\'s file (base model)', '0.9137', '—', '—'],
        ['fine-tuned, float32 (the reference)', '0.9312', '—', '—'],
        ['int8 file', '0.9314', '−0.0002', 'passed'],
        ['int4 file', '0.9276', '+0.0037', 'passed'],
      ],
      notes: [
        'The int4 cost resolved as above zero in three of the five draws; '
            'every draw\'s upper bound was under 0.01.',
        'At ten examples per option the files gained +0.0177 (int8) and '
            '+0.0138 (int4) over Google\'s file. With a linear layer fitted on '
            'every training row, Google\'s file scored higher than the '
            'fine-tune: 0.9534 against 0.9389.',
      ],
    ),
    (
      title: 'CLINC150, from the published package',
      conditions:
          '150 options. The same stages from litetune 0.7.0 as installed from '
          'PyPI, on prepare\'s split: 13,650 training and 3,900 held-out rows. '
          'The same decider and runtime.',
      header: ['embeddings', '10-shot accuracy', 'conversion cost', 'verify'],
      rows: [
        ['Google\'s file (base model)', '0.9493', '—', '—'],
        ['fine-tuned, float32 (the reference)', '0.9791', '—', '—'],
        ['int8 file', '0.9792', '−0.0001', 'passed'],
        ['int4 file', '0.9765', '+0.0026', 'passed'],
      ],
      notes: [
        'Every draw\'s upper bound was under 0.01. The files gained +0.0299 '
            '(int8) and +0.0272 (int4) over Google\'s file at ten examples per '
            'option.',
      ],
    ),
    (
      title: 'The CPU and the GPU on a Mac',
      conditions:
          'verify on the banking77 int8 file, on one Mac, with LiteRT-LM 0.18.0 '
          'from PyPI, on the CPU and with --backend gpu.',
      header: ['', 'CPU', 'GPU'],
      rows: [
        ['verify', 'passed', 'passed'],
        ['the file, 10-shot', '0.9317', '0.9311'],
        ['the reference, 10-shot', '0.9312', '0.9312'],
        ['cosine to the reference, mean', '0.99928', '0.9969'],
        ['median per text', '43.43 ms', '11.71 ms'],
      ],
      notes: [
        'The GPU computes in the precision LiteRT-LM picks, F16 unless the '
            'file declares otherwise, and its vectors sit further from the '
            'reference than the CPU\'s.',
      ],
    ),
    (
      title: 'On a Qualcomm NPU',
      conditions:
          'Measured once, on 2026-10-31. The banking77 int8 file with its text '
          'encoder compiled ahead of time for SM8750, run on a Galaxy S25 Ultra '
          'through LiteRT-LM 0.18.0\'s EmbeddingEngine C API with the Qualcomm '
          'dispatch and QNN runtime 2.50.0. The same 2,617 held-out rows; the '
          'cost is against the float32 reference, paired.',
      header: ['side', '10-shot accuracy', 'cost', 'ms per text'],
      rows: [
        ['float32 reference', '0.9312', '—', '—'],
        ['NPU, the int8 file compiled for SM8750', '0.9316', '−0.0004', '9.86'],
        [
          'the phone\'s CPU, the int8 file, 4 threads',
          '0.9314',
          '−0.0002',
          '57.6',
        ],
        [
          'NPU, Google\'s SM8750 file (base model)',
          '0.9169',
          '+0.0144',
          '6.73',
        ],
      ],
      notes: [
        'The NPU file\'s cost was unresolved in 5 of 5 draws. It is 311,705,600 '
            'bytes. convert --npu is int8 only: an int4 file compiled the same '
            'way did not load on the phone\'s NPU.',
        'No text was longer than 105 tokens, so only the 128-token signature '
            'ran.',
      ],
    ),
  ];

  /// README.md, *What has and has not been measured* and *On a Qualcomm NPU*.
  static const notMeasured = <String>[
    'Option descriptions, which the app adds to an option\'s centroid.',
    'Tasks that read distances directly, retrieval among them.',
    'The GPU on Linux, where the 0.18.0 package\'s GPU engine did not start in '
        'the one container tried, and Windows.',
    'The app\'s own runtime build, and a phone\'s CPU through the app.',
    'Training anywhere but the CPU.',
    'An NPU file for any SoC but SM8750 on a device.',
  ];

  @override
  Component build(BuildContext context) {
    return div(classes: 'emb', [
      MeasuredCards(
        page: '/models/embeddings',
        label: 'Embedding models measured end to end so far',
        runs: measured,
        note: [
          Component.text(summary),
          a(href: '/measurements#${embeddingGemma2.anchor}', [
            Component.text('Every run, with the abstention it measured'),
          ]),
        ],
      ),
      section(classes: 'emb-block', [
        h2(classes: 'emb-heading', [Component.text('What litetune does')]),
        div(classes: 'emb-steps', [
          for (final step in steps)
            div(classes: 'emb-step', [
              span(classes: 'emb-command', [
                Component.text('litetune ${step.command}'),
              ]),
              p(classes: 'emb-text', [Component.text(step.text)]),
            ]),
        ]),
      ]),
      section(classes: 'emb-block', [
        h2(classes: 'emb-heading', [Component.text('What was measured')]),
        for (final table in tables) _table(table),
      ]),
      section(classes: 'emb-block', [
        h2(classes: 'emb-heading', [Component.text('What was not')]),
        ul(classes: 'emb-list', [
          for (final item in notMeasured) li([Component.text(item)]),
        ]),
      ]),
    ]);
  }

  static Component _table(MeasuredTable t) => div(classes: 'emb-table-block', [
    h3(classes: 'emb-table-title', [Component.text(t.title)]),
    p(classes: 'emb-conditions', [Component.text(t.conditions)]),
    table(classes: 'emb-table', [
      thead([
        tr([
          for (final cell in t.header) th([Component.text(cell)]),
        ]),
      ]),
      tbody([
        for (final row in t.rows)
          tr([
            for (final cell in row) td([Component.text(cell)]),
          ]),
      ]),
    ]),
    for (final note in t.notes) p(classes: 'emb-note', [Component.text(note)]),
  ]);

  // `emb-` on every class, for the reason `measured_cards.dart` gives for its
  // own prefix: jaspr collects every component's rules into one stylesheet.
  @css
  static List<StyleRule> get styles => [
    css('.emb').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(3.rem),
    ),
    css('.emb-block').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.rem),
      raw: const {'max-width': '80ch'},
    ),
    css('.emb-heading').styles(
      fontFamily: Brand.fontMono,
      fontSize: 1.1.rem,
      fontWeight: FontWeight.w500,
      color: Brand.ink,
      margin: Margin.zero,
      padding: Padding.only(top: 1.4.rem),
      border: Border.only(
        top: BorderSide(
          color: Brand.line,
          width: 1.px,
          style: BorderStyle.solid,
        ),
      ),
    ),
    css('.emb-steps').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.rem),
    ),
    css('.emb-step').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.3.rem),
    ),
    css(
      '.emb-command',
    ).styles(fontFamily: Brand.fontMono, fontSize: 0.9.rem, color: Brand.ink),
    css('.emb-text, .emb-conditions, .emb-note, .emb-list').styles(
      color: Brand.body,
      fontSize: 0.95.rem,
      margin: Margin.zero,
      raw: const {'line-height': '1.6'},
    ),
    css('.emb-note').styles(color: Brand.muted, fontSize: 0.9.rem),
    css('.emb-list').styles(padding: Padding.only(left: 1.2.rem)),
    css('.emb-table-block').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(0.6.rem),
      margin: Margin.only(bottom: 1.2.rem),
    ),
    css('.emb-table-title').styles(
      fontSize: 1.rem,
      fontWeight: FontWeight.w600,
      color: Brand.ink,
      margin: Margin.zero,
    ),
    // Scrolls on its own at phone width rather than wrapping a number, as the
    // measurements page's tables do.
    css('.emb-table').styles(
      fontSize: 0.85.rem,
      color: Brand.body,
      raw: const {
        'border-collapse': 'collapse',
        'display': 'block',
        'overflow-x': 'auto',
        'max-width': '100%',
      },
    ),
    css('.emb-table th, .emb-table td').styles(
      padding: Padding.symmetric(horizontal: 0.6.rem, vertical: 0.4.rem),
      raw: const {
        'border': '1px solid var(--lt-line)',
        'text-align': 'left',
        'white-space': 'nowrap',
      },
    ),
    css('.emb-table th').styles(color: Brand.ink, fontWeight: FontWeight.w500),
  ];
}
