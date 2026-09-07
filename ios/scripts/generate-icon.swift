// From the repository root:
// swiftc ios/BetaBuddy/Views/BrandMark.swift ios/scripts/generate-icon.swift -o /tmp/betabuddy-icon
// /tmp/betabuddy-icon ios/BetaBuddy/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon.png
import SwiftUI
import ImageIO
import UniformTypeIdentifiers

@main
struct IconGenerator {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { fatalError("Supply an output PNG path") }
        let context = CGContext(data: nil, width: 1024, height: 1024,
                                bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, components: [0.82, 0.95, 0.29, 1])!)
        context.fill(CGRect(x: 0, y: 0, width: 1024, height: 1024))
        context.translateBy(x: 0, y: 1024)
        context.scaleBy(x: 1, y: -1)
        context.setFillColor(CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, components: [0.10, 0.12, 0.10, 1])!)
        context.addPath(BrandMark().path(in: CGRect(x: 172, y: 172, width: 680, height: 680)).cgPath)
        context.fillPath()
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        guard CGImageDestinationFinalize(destination) else { fatalError("PNG export failed") }
    }
}
