// dart format off
// ignore_for_file: type=lint

// GENERATED FILE, DO NOT MODIFY
// Generated with jaspr_builder

import 'package:jaspr/server.dart';
import 'package:litetune_website/landing/sections/formats.dart' as _formats;
import 'package:litetune_website/landing/sections/hero.dart' as _hero;
import 'package:litetune_website/landing/sections/nav_bar.dart' as _nav_bar;
import 'package:litetune_website/landing/sections/runtimes.dart' as _runtimes;
import 'package:litetune_website/landing/sections/site_footer.dart'
    as _site_footer;
import 'package:litetune_website/landing/sections/what_it_does.dart'
    as _what_it_does;
import 'package:litetune_website/landing/sections/where_to_run.dart'
    as _where_to_run;
import 'package:litetune_website/landing/sections/why_it_exists.dart'
    as _why_it_exists;
import 'package:litetune_website/landing/landing_page.dart' as _landing_page;

/// Default [ServerOptions] for use with your Jaspr project.
///
/// Use this to initialize Jaspr **before** calling [runApp].
///
/// Example:
/// ```dart
/// import 'main.server.options.dart';
///
/// void main() {
///   Jaspr.initializeApp(
///     options: defaultServerOptions,
///   );
///
///   runApp(...);
/// }
/// ```
ServerOptions get defaultServerOptions => ServerOptions(
  clientId: 'main.client.dart.js',
  styles: () => [
    ..._landing_page.LandingPage.styles,
    ..._formats.Formats.styles,
    ..._hero.Hero.styles,
    ..._nav_bar.NavBar.styles,
    ..._runtimes.RuntimesStrip.styles,
    ..._site_footer.SiteFooter.styles,
    ..._what_it_does.WhatItDoes.styles,
    ..._where_to_run.WhereToRun.styles,
    ..._why_it_exists.WhyItExists.styles,
  ],
);
