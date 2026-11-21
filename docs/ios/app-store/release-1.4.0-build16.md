# iOS 1.4.0 (16) · App Store release

Build 16 starts from the same source and assets as TestFlight build 15. It includes save-first video import, optional pose analysis that can continue while navigating, recording time read from video metadata with an edit option, and the updated offline Gym 3D model and simulation (GymPack 1.3.9). This build extends the video duration limit from 3 to 20 minutes while keeping the 500 MB file limit. The bundled pose model is unchanged.

## Verification

- 11 Swift package tests passed.
- Xcode unit and UI test suites passed, including a new test that accepts a 20-minute video and rejects one second beyond the limit.
- Signed Release archive succeeded. Bundle ID `com.betabuddy.ios`, version `1.4.0 (16)`, GymPack 1.3.9 with all 10 manifest file SHA-256 hashes matching, and the pose model file were verified in the archive.
- True-device performance, background execution duration, thermal behavior, and real camera metadata variants remain outside automated coverage.

Distribution status is tracked in [STATUS.md](STATUS.md).
