/// `/llms.txt` and `/llms-full.txt`: the site as plain text for language
/// models, in the shape llmstxt.org describes -- an H1, a one-line blockquote,
/// then sections of links.
///
/// Built from what the pages themselves render, not written out a second time:
/// the stages from `WhatItDoes`, the families from `Formats`, the measured runs
/// from the cards on `/models`, the install commands from the hero. A model
/// added to the cards is a line added here, by the same build.
library;

import 'package:jaspr/server.dart';

import 'landing/sections/formats.dart';
import 'landing/sections/hero.dart';
import 'landing/sections/quick_start.dart';
import 'landing/sections/what_it_does.dart';
import 'models/embedding_models.dart';
import 'models/llm_models.dart';
import 'models/measured_cards.dart';
import 'models/models_page.dart';
import 'seo.dart';

const String kLlmsTxtPath = '/llms.txt';
const String kLlmsFullTxtPath = '/llms-full.txt';

/// The README on GitHub, which is also what PyPI shows as the description.
const String kReadmeUrl = '$kRepositoryUrl/blob/main/README.md';

/// `/llms.txt` for the package at [version].
String llmsTxt({required String version}) {
  String run(MeasuredRun model) => '- ${model.name} (`${model.hubId}`)';

  return [
    '# litetune',
    '',
    '> $kLandingDescription',
    '',
    'litetune is a Python command-line tool, version $version: '
        '${Hero.compatibility}. Its output is a .litertlm file, the format '
        "LiteRT-LM, Google's on-device runtime, loads.",
    '',
    '## Pipeline',
    '',
    for (final (command, gloss) in WhatItDoes.steps) '- `$command`: $gloss',
    '',
    '## Supported models',
    '',
    '- Families `convert` exports: ${Formats.models}.',
    // `tune.METHODS`; the embedding recipe refuses `--method`
    // (`cli._LLM_ONLY_TUNE_FLAGS`).
    '- Fine-tuning: a language model takes `tune --method full` (the default) '
        'or `--method lora`; EmbeddingGemma 2 is a full fine-tune with a '
        'fixed recipe.',
    '',
    'Converted and compared with their float references end to end:',
    '',
    for (final model in LlmModels.measured) run(model),
    for (final model in EmbeddingModels.measured) run(model),
    '',
    '## Install',
    '',
    '```',
    ...Hero.installCommands,
    '```',
    '',
    '## ${QuickStart.heading}',
    '',
    '```',
    ...QuickStart.commands,
    '```',
    '',
    '## Docs',
    '',
    '- [README]($kReadmeUrl): install, the whole walk-through with every flag, '
        'recipes, limitations and exit codes',
    '- [Measurements]($kSiteOrigin/measurements): $kMeasurementsDescription',
    '- [Language models]($kSiteOrigin${ModelsTab.llm.path}): '
        '$kModelsDescription',
    '- [Embedding models]($kSiteOrigin${ModelsTab.embeddings.path}): '
        '$kEmbeddingsDescription',
    '- [Changelog]($kSiteOrigin/changelog): $kChangelogDescription',
    '',
    '## Optional',
    '',
    '- [Full text]($kSiteOrigin$kLlmsFullTxtPath): README.md and '
        'MEASUREMENTS.md in one file',
    '- [Source]($kRepositoryUrl): the GitHub repository',
    '- [PyPI]($kPypiUrl): the published package',
    '',
  ].join('\n');
}

/// `/llms-full.txt`: the README and the measurements, as written.
String llmsFullTxt({required String readme, required String measurements}) =>
    '$readme\n\n$measurements';

/// A route's whole response as plain text instead of a rendered page.
///
/// `jaspr build` writes a route whose path has an extension to that file name
/// rather than to `index.html` (jaspr_cli 0.23.1 `build_command.dart:320-325`), so
/// a route at `/llms.txt` comes out as `build/jaspr/llms.txt`.
/// `jaspr-sitemap-data: false` keeps it out of `sitemap.xml`, which lists
/// pages (`build_command.dart:331-333`).
class TextFile extends StatelessComponent {
  const TextFile(this.text, {super.key});

  final String text;

  @override
  Component build(BuildContext context) {
    context.setHeader('Content-Type', 'text/plain; charset=utf-8');
    context.setHeader('jaspr-sitemap-data', 'false');
    context.setStatusCode(200, responseBody: text);
    return const Component.empty();
  }
}
