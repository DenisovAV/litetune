// Stands in for package:flutter/foundation.dart in this scratch package: the
// copied decider closure names three of its members, all outside arithmetic
// (lib/src/utils/edge_ai_log.dart: kDebugMode, debugPrint;
// lib/src/lifecycle/close_notifier.dart: VoidCallback).
typedef VoidCallback = void Function();

const bool kDebugMode = true;

void debugPrint(String? message, {int? wrapWidth}) => print(message);
