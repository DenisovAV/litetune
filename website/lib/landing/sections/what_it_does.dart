import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../../theme/brand.dart';

/// The five commands, in the order they are run.
///
/// They are listed rather than shown as a terminal transcript: a transcript
/// implies a copy-pasteable happy path, and each of these takes flags that
/// depend on the model and the task. The README is where the full invocations
/// belong.
class WhatItDoes extends StatelessComponent {
  const WhatItDoes({super.key});

  @override
  Component build(BuildContext context) {
    return section(classes: 'row', [
      div(classes: 'label', [Component.text('What it does')]),
      div(classes: 'defs', [
        _step('prepare', 'splits your data and drops the rows it cannot score'),
        _step(
          'tune',
          'fine-tunes it, with the settings the export step will need',
        ),
        _step(
          'convert',
          'turns the result into the file the phone runtime loads',
        ),
        _step('verify', 'compares that file with the model it came from'),
        _step('bundle', 'packages it together with what the comparison found'),
      ]),
    ]);
  }

  static Component _step(String command, String gloss) => div(classes: 'def', [
    div(classes: 'def-term def-command', [Component.text(command)]),
    div(classes: 'def-value def-gloss', [Component.text(gloss)]),
  ]);

  @css
  static List<StyleRule> get styles => [
    // The command name is the emphasised half here, inverting the formats
    // table where the term is the quiet one — these are things you type.
    css(
      '.def-command',
    ).styles(fontFamily: Brand.fontMono, color: Brand.ink, fontSize: 1.rem),
    css('.def-gloss').styles(color: Brand.muted),
  ];
}
