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

The Firebase project id is not in this repository — it is an infrastructure
identifier, and the static site does not otherwise carry it. `deploy.sh` reads
it from the environment and fails immediately if it is unset:

```bash
LITETUNE_FIREBASE_PROJECT=<project-id> ./deploy.sh
```

The hosting site defaults to `litetune`; `LITETUNE_FIREBASE_SITE` overrides it,
and `firebase.json` names the same target. `deploy.sh` applies the target
mapping on each run, so a fresh checkout needs no `.firebaserc`; the one it
writes is git-ignored, and `.firebaserc.example` shows the shape if you would
rather commit a mapping of your own.
