# Third-party resources

- **MediaPipe Tasks Vision / Common 1.0.0** — Google, Apache-2.0. Official distributions: https://dl.google.com/cpdc/20260727-225049/MediaPipeTasksVision-1.0.0.tar.gz and https://dl.google.com/cpdc/20260727-225047/MediaPipeTasksCommon-1.0.0.tar.gz. Checksums are pinned in `scripts/bootstrap.sh`. The graph logger factory is replaced by the upstream no-op implementation; see `LocalMediaPipe/README.md`.
- **Pose Landmarker Full float16, version 1** — Google MediaPipe model bundle: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task. SHA-256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`. Unmodified weights. See https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker and the distribution's license notices.
- **MediaPipe / Abseil / glog headers** — pinned sources in `LocalMediaPipe/README.md`, used only to compile the official no-op logger against its public interface. Licenses packaged in `BetaBuddy/Resources/Licenses`.
- **Test fixture pose.jpg** — Google MediaPipe sample asset at https://storage.googleapis.com/mediapipe-assets/pose.jpg. Used only in the test bundle for inference smoke testing; not displayed as a user's climb or used as a climbing-accuracy benchmark.
- **SF Symbols** — Apple system symbols used by the native app. The climbing poster layout and BetaBuddy icon are original code-drawn compositions.
- **Gym directory coordinates (`BetaBuddy/Resources/gyms-cn.json`: `lat`, `lon`, `coordPrecision`)** — derived from OpenStreetMap through Nominatim and the Overpass API (the gym's own feature, a named building, the metro station the directory source lists as nearest, or a district / city centre, labelled per gym), © OpenStreetMap contributors, Open Database License 1.0: https://www.openstreetmap.org/copyright. The coordinate fields and the cached answers in `data/gym-directory/` are covered by ODbL, not by this project's MIT license; any screen that uses them has to credit OpenStreetMap. Gym names and addresses come from the directory's other sources; see `docs/gym-directory.md`.

## Optional local library image encoder

The camera-library development branch can bundle the image encoder from
[`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M),
revision `a2a8c6eaa2549ad66eb7c31b85022bf58273a26c`, under MIT. Attribution is in
`BetaBuddy/Resources/Licenses/TinyCLIP.txt`; upstream implementation is
[Microsoft Cream/TinyCLIP](https://github.com/microsoft/Cream/tree/main/TinyCLIP).

The private build tool exports a Core ML image encoder and includes its source-file SHA-256
values in the private classifier configuration. `scripts/install-library-model.sh` checks
all those hashes before compiling a locally supplied `.mlpackage` into ignored `Vendor/`.
The app performs generic inference with Apple's Core ML / Vision frameworks. It never
loads remote model code or downloads model files. No additional runtime SDK is linked.
Classification policy and prototypes stay in the private pack. The current candidate has
not completed independent accuracy or real-phone performance validation.
