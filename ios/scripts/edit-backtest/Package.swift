// swift-tools-version: 5.9
import PackageDescription

// A Mac command-line backtest of the app's own edit renderer. `Sources/edit-backtest/EditPlan.swift`
// and `EditRenderer.swift` are links to the app's files, not copies; only `main.swift` lives here.
// Not part of the app and not part of `swift test --package-path ios`.
let package = Package(
    name: "edit-backtest",
    platforms: [.macOS(.v14)],
    targets: [.executableTarget(name: "edit-backtest", path: "Sources/edit-backtest")]
)
