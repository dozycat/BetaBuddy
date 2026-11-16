# BetaBuddy HarmonyOS

This directory starts the native HarmonyOS 5+ port. The portable geometry and import limits in
`entry/src/main/ets/core/` match the iOS and Android record model. They have not yet been compiled
with DevEco Studio; this directory is not an installable app.

The application will use a Stage-model ArkTS project. The two product entries remain Gym and
Logbook. GymPack's HTML/JavaScript and offline assets can be evaluated with ArkWeb. Video selection,
private storage, playback, frame decoding, pose inference, and background processing need native
HarmonyOS implementations. The Android MediaPipe AAR, MediaCodec, Media3, and foreground service
cannot run in a HarmonyOS 5+ HAP.

Start with DevEco Studio 26.0.0 Release's **Application → Empty Ability** template. Set the compatible
SDK to the oldest version that supports the required APIs and the intended HarmonyOS 5+ devices;
the template guide's API 26 example would narrow device coverage. Add the code in this directory
to `entry/src/main/ets/core/`, then build and run on a HarmonyOS 5+ device. The next
implementation slice is a local video import, saved record, and playback; pose analysis follows
after a device-tested inference runtime is chosen. A release build must also include the checked
GymPack snapshot and the platform's purchase integration before it is submitted to AppGallery.

Official starting point: https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/start-with-ets-stage/
