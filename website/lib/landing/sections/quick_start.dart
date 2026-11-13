import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The first four commands of the README's walk-through, as text a reader can
/// copy and a search engine can read.
///
/// Copied, not paraphrased: `test/seo_test.dart` requires every command line
/// of [commands] to be a line of README.md, and every comment to begin one, so
/// a flag renamed there fails the site's tests instead of leaving this block
/// quietly wrong. The README's comments explain each step at length; the ones
/// here are their opening sentences. `bundle`, the fifth step, needs inputs this block does
/// not introduce, so the link to the README is where it is found.
class QuickStart extends StatelessComponent {
  const QuickStart({super.key});

  static const heading = 'How to fine-tune a model for LiteRT-LM';

  /// Where the full walk-through is, `bundle` and every flag included.
  static const readmeUrl =
      'https://github.com/DenisovAV/litetune#from-your-data-to-a-shippable-bundle';

  static const intro =
      'The README\'s example, prepare → tune → convert → verify: FunctionGemma '
      '270M fine-tuned with LoRA (leave out --method for a full fine-tune, the '
      'default), converted to .litertlm with two quantization recipes, and '
      'compared with the float model on the held-out split.';

  static const commands = [
    '# 1. Validate and split the data. Seconds.',
    r'litetune prepare --data raw.jsonl --output-dir data --context-length 1024 \',
    '                 --tokenizer google/functiongemma-270m-it',
    '',
    '# 2. Fine-tune. Chooses CUDA, then MPS on a Mac, then CPU.',
    r'litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \',
    '              --output-dir tuned --method lora --prompt-mode prerendered',
    '',
    '# 3. Convert, sweeping recipes rather than trusting a default.',
    r'litetune convert --model tuned/model --output-dir artifacts \',
    '                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32',
    '',
    '# 4. Compare the converted model with its float twin.',
    r'litetune verify --model artifacts/weight_only_wi8_afp32/<name>.litertlm \',
    r'                --reference tuned/model --data data/heldout.jsonl \',
    '                --json > manifest.json',
  ];

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      h2(classes: 'label', [Component.text(heading)]),
      div(classes: 'quick', [
        p(classes: 'quick-intro', [Component.text(intro)]),
        pre(classes: 'quick-code', [
          code([Component.text(commands.join('\n'))]),
        ]),
        p(classes: 'quick-more', [
          a(
            href: readmeUrl,
            attributes: const {'target': '_blank', 'rel': 'noopener'},
            [Component.text('Every flag, and the bundle step, in the README')],
          ),
        ]),
      ]),
    ]);
  }

  @css
  static List<StyleRule> get styles => [
    css('.quick').styles(
      display: Display.flex,
      flexDirection: FlexDirection.column,
      gap: Gap.all(1.rem),
      // `min-width: 0` so the code block scrolls inside the column instead of
      // widening it: a flex item's minimum width is its content's by default.
      raw: const {'max-width': '62ch', 'min-width': '0'},
    ),
    css('.quick-intro').styles(
      color: Brand.body,
      fontSize: 1.rem,
      lineHeight: 1.6.em,
      margin: Margin.zero,
    ),
    css('.quick-code').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.8.rem,
      lineHeight: 1.6.em,
      color: Brand.ink,
      padding: Padding.all(1.rem),
      margin: Margin.zero,
      backgroundColor: Brand.surface,
      border: Border.all(color: Brand.line, width: 1.px),
      radius: BorderRadius.circular(0.6.rem),
      overflow: Overflow.auto,
    ),
    css('.quick-code code').styles(fontFamily: Brand.fontMono),
    css('.quick-more').styles(
      fontFamily: Brand.fontMono,
      fontSize: 0.85.rem,
      margin: Margin.zero,
    ),
    css(
      '.quick-more a',
    ).styles(color: Brand.ink, raw: const {'text-underline-offset': '3px'}),
  ];
}
