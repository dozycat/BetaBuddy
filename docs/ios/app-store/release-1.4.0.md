# 1.4.0 (15) · Save first, optional background analysis

## TestFlight — What to Test

- Import now saves the original video by default, without waiting for pose analysis. Check playback, notes, and send status immediately after saving.
- Recording time is read from video metadata when available. You can edit the date and time during import or later in Organize this climb. Missing metadata falls back to the current time with an explanation.
- Turn on Analyze skeleton after saving, or start analysis from a saved record. Leave the page, switch tabs, or briefly background the app; the saved video and notes remain available.
- iOS limits background execution time. If that allowance expires, the pending request is kept and analysis restarts from the beginning on the next foreground or launch. Explicitly stopping removes the request.
- Check that changes to date, grade, and notes made during analysis are kept when the result arrives.

Everything runs locally. No account or cloud upload is required. Gym content remains unofficial and modeled movement is not coaching advice.

## Build and validation

- iOS version 1.4.0, build 15; bundled GymPack 1.3.9, matching the installed local package. Package files were verified against the manifest without committing pack contents.
- 11 Swift core tests passed. 39 iOS unit/integration tests: 38 passed, 1 optional local-video test skipped. Existing 5 UI tests and the new import-defaults UI test passed.
- Unsigned device build and signed Release archive succeeded. True-device background duration, thermals, and camera/cloud metadata variants remain unverified.
- Distribution status is tracked in [STATUS.md](STATUS.md). This release is intended for TestFlight first, not an App Store review submission.
