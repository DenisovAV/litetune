# litetune website

The marketing site for [`litetune`](https://github.com/DenisovAV/litetune),
built with [Jaspr](https://jaspr.site) in static mode and modelled on the
`flutter_gemma` website in this author's other repository.

```bash
dart pub get
jaspr serve          # http://localhost:8080, hot reload
jaspr build          # static output in build/jaspr
./deploy.sh          # build + deploy to Firebase Hosting
```

## Layout

| Path | What it holds |
|---|---|
| `lib/main.server.dart` | The only entrypoint that matters — renders one `Document` at `/` |
| `lib/seo.dart` | Open Graph / Twitter / canonical / JSON-LD head tags |
| `lib/theme/brand.dart` | Design tokens as CSS custom properties, dark by default |
| `lib/landing/landing_page.dart` | Composes the sections |
| `lib/landing/sections/` | One file per section |

## Theming

The page is **dark by default** and flips to light under
`prefers-color-scheme: light`. Both themes are monochrome: there is no accent
colour, so emphasis is carried by weight, size and rules rather than hue.
Every colour is a CSS custom property declared once in `brand.dart`; nothing
else in the codebase hardcodes a hex value.

## Before the first deploy

`deploy.sh` and `firebase.json` carry a placeholder Firebase project and
hosting target. Fill both in, or replace them with whatever host ends up
serving `litetune.dev`.
