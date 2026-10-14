# Goldens for `litetune decide`, written by flutter_edge_ai's own Dart code

Every expected number here was computed by the decider code of flutter_edge_ai, not by
litetune: `tests/test_decide.py` holds litetune's fit to them.

## Source

- Repository: DenisovAV/flutter_gemma, branch `feat/decision-engine`
- Commit: `77f0484e74d91b19c9c574c7ec58b7103a76f643` (2026-10-14, "test(decision): fit
  fingerprints allow the platform libm's last bits in the linear layer"), flutter_edge_ai 2.4.0
- Run with the Dart SDK 3.13.3 (stable) on macos_arm64, on 2026-10-14.

`generator/goldens.dart` wrote every JSON file here. It ran in a scratch Dart package
named `flutter_edge_ai` whose `lib/` held these 33 files of
`packages/flutter_edge_ai/lib/` copied unchanged from that commit (sha256 of each, equal to
the committed blob):

```
3d49b1c8122c13efb9feaf3c420da4c45ba2a5e7c684f2aaa73352a6a5e32b6f  src/decision/decider.dart
8c037a7006ae3bafb432f45ec004ec14dac47a4241f6bfe84d85b87071ab1c4a  src/decision/decider_embedding.dart
01bea46cc0b2d9c7d67fb7088aa9266b422bbda19bdf7a8cb7c5a08452143c0a  src/decision/decider_exceptions.dart
ca11b7f1ba1a53983f15f99b485930d139c19898b284a4653db56363b7662170  src/decision/decider_fit.dart
12981d78caaf5dac950df81f2f01403196755a2de53201b5cd96cd5ee8f3beca  src/decision/decider_method.dart
4081f37364e775bd22de1915d00f8cb564e40792472da1e755d66377b01cb40a  src/decision/decider_option.dart
7526a121d918aff03ac734cd7339f69824a698837202daccdfda01cb134abb70  src/decision/dense_kernel.dart
aa41bea35c0611979ae3f40a63b882844d5fbfbc2afebaedd94dcdaa7780f8b0  src/decision/embedding_decider.dart
37134807eb26f004ddebf6a2021dd32e4ac76c8eeab67edf879c255443f918aa  src/decision/fit_pacer.dart
1e06867636aefde98353a491d3fb5e5404c96b51372aeea390f5a90b842f8a57  src/decision/fit_runner_io.dart
b3611f0c78acdb290ccb9d534936742c0ebe1194f85d5539fc86577631633b97  src/decision/fit_runner_web.dart
11cd1c54a767bd2b7458afa16c22037687fb2f46eec8185b34c6795035c7df2c  src/decision/softmax_regression.dart
f276e11821ce754eef2d1073d8b194fec97b228abcbfba337d63c5480b0a8cb6  src/domain/platform_types.dart
200d48267589d22392d85f5b2fd0f203fdc80a6900b1f924b601c1ca2c66f57a  src/embedding/common_embedding_model.dart
cebaa0b06eafcab68f692f69ca2635aeb8f60824cd04f39c5ab6e1f908bc723e  src/embedding/embedding_exceptions.dart
e1f59b421fde17aa8cbab267a43eec953793d79d62950dc49fbbc3194ba4f95d  src/embedding/embedding_host.dart
8f132d406b12b2e81f681f9dd432f99c518a0059b72b5c772ecdb6986ee10826  src/embedding/embedding_host_io.dart
3e575ad099acaa3342b2f1e7ba775e6a12a31203d2c5b6fcdfd109fd8dfe5b71  src/embedding/embedding_host_web.dart
d44569f6571306523afcad00a69f13df8117680d99ed79771a4c6ee0fe77b774  src/embedding/embedding_input.dart
fe242046fb08769aab10c504d9b4e1331766b69b9f24a6e703851da1619e77da  src/embedding/embedding_modality.dart
cf84efc9429ae6b3ddd856be67bc7491e436be1ff0aeaeb06a43c62878b5bc33  src/embedding/embedding_model.dart
10494edddb8d3a7c38b0e942864f59a120c7056d4395491ff483552127d22d94  src/embedding/embedding_pass.dart
6104acbbccca8aafc8a5eb2d1985cd099fcb517e34fcaa7f4a5e5c77221db082  src/embedding/event_loop_yield.dart
7d262f958ca6598e5fb5a3089a36372fbedf292204e911fa603df2f078352418  src/embedding/event_loop_yield_web.dart
624e6dbb32d69332b57536acfeaccc99953924608577e4d7f5e0bdc38b672c4b  src/embedding/load_check.dart
d80535022aa00d4a562ffa107c342c6641e8bda6d853033d8824c081b269c86f  src/embedding/request_lanes.dart
60e396e381087f654141aa4b3157d92b3f069b6ce74e9d57f649e9ddb220d9fc  src/embedding/task_type.dart
9c68447695b4956aa9acbec01b358aecf8d4aedf44aacad771c168fcc2b1bc9b  src/embedding/wire.dart
3dee55a47c038506cabb921f482d4d107f9ec7ac483cd591b231721bb86135d1  src/lifecycle/close_notifier.dart
1d1ecd9dcced6e48dbc1c982eab85cf6b698cad0db2a5448a4b95dc66a5f3c31  src/media/encoded_media_format.dart
0e1d57f831e863e298d508a221a17b9492e013b1af62e801742623cc0a6b7876  src/message.dart
af5982c33357e05b66342ebde93d3dcbd47888a4b797a30c2720fa038a88c97f  src/model_management/cancel_token.dart
2ce4968572e9258ed804e1bd267b30b2544ea988f2c7ce8786231d99987f2827  src/utils/edge_ai_log.dart
```

Two of them import `package:flutter/foundation.dart` for a log call and a callback type
(`utils/edge_ai_log.dart`, `lifecycle/close_notifier.dart`); `generator/shim_flutter/`
stands in for it with those three names, outside any arithmetic. `generator/pubspec.yaml`
is the package's pubspec. Run from that package: `dart run bin/goldens.dart <out dir>`;
two runs wrote byte-identical files.

## What each file holds

The cases go through the path an app takes: a `CommonEmbeddingModel` with the CPU backend
and the installed file `embeddinggemma-2-text-270m.litertlm` of 164,626,432 bytes, serving a
fixed table of raw float32 vectors (`vectors-<group>.json`); `createDecider`; `toArtifact()`
through `jsonEncode`/`jsonDecode` (`artifact`); `choose` on each probe with every option
ranked (`probes`), checked against a decider `loadDecider` rebuilt from the artifact.
`internals` holds the fit's own functions run on the unit rows `createDecider` builds:
`optionCentroids` (float64), `crossFitSignal`, and `fitSoftmaxRegression`'s float64
weights, bias, loss, iterations and evaluations. A case the Dart code refuses has `refused`
with its error type and message instead.

| File | What it exercises |
|---|---|
| `small-*.json` | 4 options, 16 dimensions, two with a description (7 inputs, two uneven folds), two without (6), metadata on one; both methods; crossFit 0.95 and 0.8, labelled (20 rows, 4 labelled as the next option), minSimilarity 0.25, none; 12 probes |
| `keep09-ten-inputs.json` | keep 0.9 over ten held-out inputs, where `(1 - 0.9) * 10` is 0.9999999999999998 |
| `ties-*.json` | two labelled rows sharing one vector, one right and one wrong, at the lowest signal: maxError 0.05 excludes the group whole, 0.25 admits it; two identical examples in one option |
| `overlap-*.json` | 3 options of 7, 6 and 8 inputs with four training examples drawn near another option, so some held-out inputs' best centroid is another option's fold-centroid; 36 labelled rows, 12 of them between two options, some of which `linear` and the centroids answer differently; both methods, crossFit and labelled (maxError 0.3); 82 L-BFGS iterations |
| `refused-*.json` | the temperature at either end of its grid, no threshold within maxError 0, two inputs that cancel out |
| `fingerprints.json` | `test/decision/fit_fingerprint.dart`'s model-free fixture (10, 50, 150 options x 10 rows, 768 dimensions, crossFit 0.95, both methods): the fingerprints, equal to that file's `vmFingerprint*` constants, and for 10 options the full centroids, weights, bias and cross-fit signal |
| `lbfgs.json` | `minimizeLbfgs` itself on four polynomial objectives the generator defines (`rosenbrock`, `quartic`, `well`, `mix`): Rosenbrock from its defaults and with `maxIter` 5, `maxEval` 12 and 20, `historySize` 3, `toleranceChange` 1e-3, `toleranceGrad` 1e-2, `lr` 0.5, a far start and a start at the minimum; and five runs that reach what those do not -- a curvature pair below 1e-10, a line search capped by the evaluations left, the sufficient-decrease constant, the step-size stop, the zoom's insufficient-progress rule (found by changing each in litetune's port). The final point, loss, iterations and evaluations |
| `space-ids.json` | `deciderSpaceId` for the installed file, for a name with every escaped character, and for an `embedderId` |

`generator/load_litetune.dart`, run from the same package, loads an artifact litetune wrote
with that code's own `loadDecider` and answers probe vectors with `choose`: `dart run
bin/load_litetune.dart <decider.json> <probes.json> <answers.json>`, the probes a JSON map
of text to raw vector. On 2026-10-14 it loaded two artifacts litetune's fit wrote from this
fixture's 768-dimension rows (150 options, `linear`, `crossFit`; 50 options, `linear`,
`labelled`) and answered 200 probes each exactly as litetune's decision code did: the same
choice, cosine and score, to the bit.

## sha256

```
4697e86334a21528bb9659a137521e201efaf729d7afff933c62ae79680fc21f  fingerprints.json
effc650de9575b55741b41bca92aaeae74608a23fcaaa183b8cbcb0bb7d2967e  keep09-ten-inputs.json
24a8ab2e78d96fe0398cac40695cae8f6356cccc51d6c490630d43cbcbde9d83  lbfgs.json
47e3241312a958e4090d8c214eb9c9c2457a6d586d5ee00e9310110c76b77f65  overlap-centroid-crossfit.json
bd7f63e2b90be43a75d40bae17274c1c6ca393e5059d0250688a224ef5389602  overlap-centroid-labelled.json
09728f556f8006e15b3c3519a8c12f3f5bf60ec8d346742bab0c77537f36549e  overlap-linear-crossfit.json
e12eb6d78c888aa0ffd8e7bfef8936edbb653df6ecb706bd2ca3389b2a5937a2  overlap-linear-labelled.json
6ff41599d91675c87ada3510838f6f1d835db290eae9cd33d071a517dd385072  refused-cancelling-inputs.json
90c220a392504c1555d957f7b21ef81a27d8095e89d2425281b4e4be75b94bf3  refused-no-threshold.json
d20e24ded16917eab75ab399581543ac17c949b41e02ed41c86ceeebd982617b  refused-temperature-ceiling.json
66ddf2c388b99b446e7fc3058832f5a70c9c3306904f28ddfbe1de65329d8ca7  refused-temperature-floor.json
7d86a451f2ecf3dee86bd8906c89a77e02ab25bd95693de6e165d033960aacf4  small-centroid-crossfit.json
ae7b3701de732a8eac6f10f58126c8e28a31d60859d2bf794324a09ee4e61fab  small-centroid-labelled.json
450fe26eedbfaa882a3a4be351d9d44ebfca1a3051b8569435c0b2ab6ba28ad7  small-centroid-minsimilarity.json
b9f5696fb12f4ffd3afc4c1707a28a6b333227fd964edffa34436078cd40fc3c  small-linear-crossfit-keep08.json
78737430540b93fb030f33ff130a741735793388ef622797c8deab98608c3559  small-linear-crossfit.json
f65d211389c80a5d32500302bd306d6cccc67a83e6ffac79dccf76f31694a17c  small-linear-labelled.json
fc24cad00de06bd088e14ee9c37156bbd66f7459537cea32146da286d0f459fc  small-linear-none.json
5227fbfbd03d70c89238d25d99aecd30335eb01d8cc0d20c5d6cda2596dbe0df  space-ids.json
27e69c1dc5e7e63c1da2e184a1ed1aaa159746818436c13765a2aeec9ea20ff6  ties-centroid-maxerror0.05.json
6a92f5044432179c188fff85482c061ee2dc48984b0400e4dda45cc1be3aaa3e  ties-centroid-maxerror0.25.json
03060e9be20d8757669e779a7e6cc793995fd685ae2a8cceb6bb1ade74223eb9  ties-crossfit.json
504c08fad45ebe6a87c9450f237fa1854ec7f07a7288ff264ea1132224fb0a38  ties-linear-maxerror0.05.json
cb857648dca1720c832bc202cc20e36485559805dbe52fa43ecc7415b584bd75  ties-linear-maxerror0.25.json
3ac3b259c20d0bc453de11e577d458c8498aea84bdcbbb43c47e71cb48f311dc  vectors-keep09.json
3121cc36d84adce612052929371d8f1d5f121a8c060919d8e62e542766b63ac9  vectors-overlap.json
083db64c818a9a54f9a42210941a98a62004c1ac01cff7bc9dbcb5b52b566131  vectors-refusals.json
b49592654909dc7fc67f86832643a83bd38516b6137e465a9f34508c0b698545  vectors-small.json
751aa127a7e325467efc65b01c2d801f8c137a7e67d517e0ceaf489f2f5a37e4  vectors-ties.json
c8ce5042b7da0ce5ff9f1a4674210750f423064a04c812529532a5cbbcd2b4e0  generator/goldens.dart
530bdc66775dadeb5b6c9628bf3c5814608af3d908b4abbad69337029652b271  generator/load_litetune.dart
6fe215142ab13ebec3d2ad7ae2174539963e18acdc9f1b53f9df7fef81294bb2  generator/pubspec.yaml
c1ebd9b3f2a795be3494b1cb2ffa7b9d46e87e4bfcfb59613a04b29921e0f0d4  generator/shim_flutter/pubspec.yaml
de5c8655f62768a9a77f02fd65ebd316b45a958094e6336a6b400541b981c336  generator/shim_flutter/lib/foundation.dart
```
