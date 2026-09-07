// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BetaBuddyCore",
    platforms: [.macOS(.v13)],
    products: [.library(name:"BetaBuddyCore",targets:["BetaBuddyCore"])],
    targets: [
        .target(name:"BetaBuddyCore",path:"BetaBuddy/Core",exclude:["VideoAnalyzer.swift","ClimbStore.swift"],sources:["Models.swift","PoseMath.swift"]),
        .testTarget(name:"BetaBuddyCoreTests",dependencies:["BetaBuddyCore"],path:"CoreTests")
    ]
)
