import SwiftUI

/// An open climbing link, drawn on a 100 × 100 grid. No font or symbol dependency.
struct BrandMark: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: 33, y: 88))
        p.addCurve(to: CGPoint(x: 20, y: 47), control1: CGPoint(x: 16, y: 77), control2: CGPoint(x: 10, y: 63))
        p.addLine(to: CGPoint(x: 43, y: 14))
        p.addCurve(to: CGPoint(x: 75, y: 8), control1: CGPoint(x: 51, y: 2), control2: CGPoint(x: 65, y: 1))
        p.addLine(to: CGPoint(x: 88, y: 17))
        p.addCurve(to: CGPoint(x: 94, y: 49), control1: CGPoint(x: 100, y: 25), control2: CGPoint(x: 101, y: 38))
        p.addLine(to: CGPoint(x: 82, y: 66))
        p.addLine(to: CGPoint(x: 64, y: 54))
        p.addLine(to: CGPoint(x: 76, y: 37))
        p.addQuadCurve(to: CGPoint(x: 74, y: 32), control: CGPoint(x: 78, y: 34))
        p.addLine(to: CGPoint(x: 63, y: 25))
        p.addQuadCurve(to: CGPoint(x: 57, y: 26), control: CGPoint(x: 59, y: 22))
        p.addLine(to: CGPoint(x: 37, y: 56))
        p.addQuadCurve(to: CGPoint(x: 39, y: 65), control: CGPoint(x: 33, y: 61))
        p.addLine(to: CGPoint(x: 48, y: 71))
        p.addQuadCurve(to: CGPoint(x: 55, y: 69), control: CGPoint(x: 52, y: 74))
        p.addLine(to: CGPoint(x: 58, y: 65))
        p.addLine(to: CGPoint(x: 76, y: 77))
        p.addLine(to: CGPoint(x: 72, y: 83))
        p.addCurve(to: CGPoint(x: 40, y: 93), control1: CGPoint(x: 65, y: 95), control2: CGPoint(x: 52, y: 101))
        p.closeSubpath()
        let scale = min(rect.width, rect.height) / 100
        return p.applying(CGAffineTransform(scaleX: scale, y: scale))
            .offsetBy(dx: rect.midX - 50 * scale, dy: rect.midY - 50 * scale)
    }
}
