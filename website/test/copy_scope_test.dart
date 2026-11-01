import 'package:litetune_website/landing/sections/formats.dart';
import 'package:litetune_website/landing/sections/hero.dart';
import 'package:litetune_website/landing/sections/what_it_does.dart';
import 'package:litetune_website/landing/sections/where_to_run.dart';
import 'package:litetune_website/landing/sections/why_it_exists.dart';
import 'package:litetune_website/models/llm_models.dart';
import 'package:litetune_website/seo.dart';
import 'package:test/test.dart';

void main() {
  test('measurement summary keeps devices and measured subsets scoped', () {
    expect(LlmModels.measurementSummary, contains('converted candidate'));
    expect(LlmModels.measurementSummary, contains('float references'));
    expect(
      LlmModels.measurementSummary,
      contains('Four of the fine-tuned models'),
    );
    expect(
      LlmModels.measurementSummary,
      contains("candidates on LiteRT-LM's CPU backend"),
    );
    expect(LlmModels.measurementSummary, contains('phone run'));
    expect(LlmModels.measurementSummary, isNot(contains('Measured on CPU')));
    expect(
      LlmModels.measurementSummary,
      isNot(contains('Four bits cost more')),
    );
  });

  test('format copy separates runtime support from litetune execution', () {
    expect(Formats.acceleration, contains('LiteRT-LM'));
    expect(Formats.acceleration, contains('platform-specific NPU support'));
    expect(Formats.accelerationQualifier, contains('training'));
    expect(Formats.accelerationQualifier, contains('float reference'));
    expect(Formats.accelerationQualifier, contains('CPU export path'));
    expect(Formats.accelerationQualifier, contains('CPU or GPU'));
  });

  test('measured-model copy covers every card', () {
    expect(LlmModels.measured, hasLength(12));
    expect(
      LlmModels.measured.map((model) => model.name),
      contains('Qwen2.5 0.5B Instruct'),
    );
  });

  test('stage and platform summaries do not overstate their scope', () {
    expect(WhatItDoes.prepareDescription, isNot(contains('drops')));
    expect(WhatItDoes.convertDescription, contains('LiteRT-LM'));
    expect(Hero.description, contains('supported native devices'));
    expect(Hero.compatibility, contains('Windows except convert'));
    expect(kOperatingSystems, contains('Windows'));
    expect(WhereToRun.galleryNote, contains('Android, iOS and macOS'));
  });

  test(
    'the landing page counts the cards on /models rather than listing them',
    () {
      expect(WhyItExists.teaser, contains('12 language models'));
      expect(WhyItExists.teaser, contains('1 embedding model,'));
      for (final model in LlmModels.measured) {
        expect(WhyItExists.teaser, isNot(contains(model.name)));
      }
      expect(Formats.measuredQualifier, isNot(contains('Gemma 4 E2B')));
    },
  );

  test('model cards avoid recommendations not established by one run', () {
    for (final model in LlmModels.measured) {
      expect(model.what, isNot(contains('the one to try first')));
      expect(model.what, isNot(contains('the size to move to')));
    }
  });
}
