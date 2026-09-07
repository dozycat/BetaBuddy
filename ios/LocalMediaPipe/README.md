# Local-only MediaPipe logger integration

The official MediaPipe 1.0.0 distribution contains a metrics logger. BetaBuddy replaces **only its factory** with upstream's `TasksDummyLogger` (drops all events), keeping the Full model and inference graph unchanged. This avoids requiring telemetry consent to analyze a local video.

`prepare-local-graph.sh` copies the official device/simulator graph archives into ignored `Vendor/LocalGraph`, removes `logging_factory.o`, and fails if the expected factory object is absent. The app compiles `LocalTasksLogger.cc` to provide the same factory symbol using the actual upstream interface, rather than duplicating its ABI by hand. The original Vendor SDK is never edited. Bootstrap reruns this derivation. Runtime downloads are not used.

Pinned header sources (unmodified except build configuration macros in our factory):

- MediaPipe: `0ad5a71bcdff3d756dc5b07f93765aaeb4152538`, the last public commit preceding the SDK release timestamp, Apache-2.0.
- Abseil: `20260107.1`, Apache-2.0; header dependencies only, no alternate Abseil binary is linked.
- glog: `v0.7.1`, BSD-3-Clause; header dependencies only. Export annotations configured in our translation unit for static compilation.

Original license/copyright notices remain in every header. Full licenses are included in the app's `Resources/Licenses` directory. Model and SDK provenance are documented in `THIRD_PARTY.md`.

This is intentionally pinned. On an SDK update, review its logger interface and factory object, repeat model smoke tests for device and simulator, and verify the final binary resolves `CreateTasksLogger` to `TasksDummyLogger`. Do not silently fall back to the vendor factory.
