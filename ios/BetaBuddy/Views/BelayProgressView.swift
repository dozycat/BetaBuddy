import SwiftUI

/// A small rope and the app's own hold mark. Height follows reported work, never a timer.
struct BelayProgressView: View {
    var progress: Double
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var fraction: Double { progress.isFinite ? min(1, max(0, progress)) : 0 }

    var body: some View {
        ViewThatFits(in: .vertical) {
            presentation(ropeHeight: 210)
            presentation(ropeHeight: 84)
            presentation(ropeHeight: 0)
        }
        .foregroundStyle(BB.paper)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("正在剪辑…"))
        .accessibilityValue(Text(fraction, format: .percent.precision(.fractionLength(0))))
    }

    private func presentation(ropeHeight: CGFloat) -> some View {
        VStack(spacing: 18) {
            if ropeHeight > 0 {
                GeometryReader { geometry in
                    let top: CGFloat = 24
                    let bottom = geometry.size.height - 26
                    let left = geometry.size.width * 0.35
                    let right = geometry.size.width * 0.65
                    let y = bottom - (bottom - top) * fraction
                    ZStack {
                        Path { path in
                            path.move(to: CGPoint(x: left, y: bottom))
                            path.addLine(to: CGPoint(x: left, y: top))
                            path.addQuadCurve(to: CGPoint(x: right, y: top), control: CGPoint(x: geometry.size.width / 2, y: -4))
                            path.addLine(to: CGPoint(x: right, y: bottom))
                        }.stroke(BB.paper.opacity(0.28), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        Path { path in
                            path.move(to: CGPoint(x: left, y: bottom))
                            path.addLine(to: CGPoint(x: left, y: y))
                        }.stroke(BB.lime, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        Circle().fill(BB.paper).frame(width: 7, height: 7)
                            .position(x: geometry.size.width / 2, y: 10)
                        ZStack {
                            MemberHoldShape().fill(BB.lime)
                            Circle().fill(BB.ink).frame(width: 6, height: 6)
                        }.frame(width: 33, height: 35).rotationEffect(.degrees(-12))
                            .position(x: left, y: y)
                    }.animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: fraction)
                }.frame(width: 140, height: ropeHeight).accessibilityHidden(true)
            }
            Text("正在剪辑…").font(.headline)
            Text(fraction, format: .percent.precision(.fractionLength(0)))
                .font(.system(.title2, design: .rounded, weight: .bold)).monospacedDigit()
        }
    }
}
