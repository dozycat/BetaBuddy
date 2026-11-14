import SwiftUI

enum BB {
    static let paper = Color(red: 0.96, green: 0.95, blue: 0.91)
    static let ink = Color(red: 0.10, green: 0.12, blue: 0.10)
    static let lime = Color(red: 0.82, green: 0.95, blue: 0.29)
    static let lavender = Color(red: 0.82, green: 0.80, blue: 0.94)
    static let muted = Color(red: 0.38, green: 0.40, blue: 0.35)
    static let line = ink.opacity(0.13)
    static func display(_ size: CGFloat) -> Font { .system(size: size, weight: .black, design: .rounded) }
}

/// A lightly faceted silhouette inspired by a climbing hold. Its hit area stays rectangular.
struct HoldShape: Shape {
    func path(in rect: CGRect) -> Path {
        let w = rect.width, h = rect.height
        var path = Path()
        path.move(to: CGPoint(x: w * 0.07, y: h * 0.04))
        path.addQuadCurve(to: CGPoint(x: w * 0.94, y: h * 0.05), control: CGPoint(x: w * 0.49, y: 0))
        path.addQuadCurve(to: CGPoint(x: w * 0.99, y: h * 0.22), control: CGPoint(x: w, y: h * 0.07))
        path.addLine(to: CGPoint(x: w * 0.97, y: h * 0.83))
        path.addQuadCurve(to: CGPoint(x: w * 0.89, y: h * 0.98), control: CGPoint(x: w * 0.96, y: h))
        path.addQuadCurve(to: CGPoint(x: w * 0.08, y: h * 0.96), control: CGPoint(x: w * 0.50, y: h * 0.99))
        path.addQuadCurve(to: CGPoint(x: w * 0.015, y: h * 0.79), control: CGPoint(x: 0, y: h * 0.92))
        path.addLine(to: CGPoint(x: w * 0.025, y: h * 0.20))
        path.addQuadCurve(to: CGPoint(x: w * 0.07, y: h * 0.04), control: CGPoint(x: 0, y: h * 0.06))
        path.closeSubpath()
        return path
    }
}

/// The smaller member emblem uses a natural hold profile instead of stretching the button shape.
struct MemberHoldShape: Shape {
    func path(in rect: CGRect) -> Path {
        let w = rect.width, h = rect.height
        var path = Path()
        path.move(to: CGPoint(x: w * 0.13, y: h * 0.24))
        path.addCurve(to: CGPoint(x: w * 0.44, y: h * 0.06), control1: CGPoint(x: w * 0.12, y: h * 0.09), control2: CGPoint(x: w * 0.30, y: h * 0.04))
        path.addCurve(to: CGPoint(x: w * 0.87, y: h * 0.21), control1: CGPoint(x: w * 0.66, y: h * 0.01), control2: CGPoint(x: w * 0.84, y: h * 0.08))
        path.addCurve(to: CGPoint(x: w * 0.92, y: h * 0.52), control1: CGPoint(x: w * 0.98, y: h * 0.32), control2: CGPoint(x: w * 0.96, y: h * 0.39))
        path.addCurve(to: CGPoint(x: w * 0.73, y: h * 0.91), control1: CGPoint(x: w * 0.93, y: h * 0.74), control2: CGPoint(x: w * 0.88, y: h * 0.87))
        path.addCurve(to: CGPoint(x: w * 0.32, y: h * 0.92), control1: CGPoint(x: w * 0.57, y: h * 0.98), control2: CGPoint(x: w * 0.45, y: h * 0.90))
        path.addCurve(to: CGPoint(x: w * 0.06, y: h * 0.63), control1: CGPoint(x: w * 0.11, y: h * 0.95), control2: CGPoint(x: w * 0.03, y: h * 0.80))
        path.addCurve(to: CGPoint(x: w * 0.13, y: h * 0.24), control1: CGPoint(x: w * 0.02, y: h * 0.47), control2: CGPoint(x: w * 0.10, y: h * 0.37))
        path.closeSubpath()
        return path
    }
}

struct BoltMark: View {
    var color: Color = BB.ink
    var body: some View {
        ZStack {
            Circle().strokeBorder(color, lineWidth: 1.7)
            Image(systemName: "hexagon.fill").font(.system(size: 8, weight: .black)).foregroundStyle(color)
        }.frame(width: 22, height: 22).accessibilityHidden(true)
    }
}

/// Member mark: a route hold with a star set where its bolt would sit.
struct MemberMark: View {
    var body: some View {
        ZStack {
            MemberHoldShape().fill(BB.lime)
            MemberHoldShape().stroke(BB.ink.opacity(0.65), lineWidth: 1)
            Image(systemName: "star.fill")
                .resizable().scaledToFit().foregroundStyle(BB.ink)
                .padding(8)
        }.aspectRatio(1, contentMode: .fit).accessibilityHidden(true)
    }
}

struct RouteButtonStyle: ButtonStyle {
    enum Tone { case dark, lime, paper }
    var tone: Tone = .dark
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .foregroundStyle(tone == .dark ? BB.paper : BB.ink)
            .frame(minHeight: 48)
            .background(HoldShape().fill(tone == .dark ? BB.ink : tone == .lime ? BB.lime : .white.opacity(0.65)))
            .overlay(HoldShape().stroke(BB.ink.opacity(tone == .paper ? 0.16 : 0.05), lineWidth: 1))
            .contentShape(Rectangle())
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

// Stamps and the monospaced trailing labels are brand marks in English; they are not translated.
struct Stamp: View {
    var text: String
    var color: Color = BB.ink
    var body: some View {
        Text(verbatim: text).font(.system(size: 10, weight: .bold, design: .monospaced)).tracking(1.3).lineLimit(1).minimumScaleFactor(0.7)
            .padding(.horizontal, 9).padding(.vertical, 6)
            .foregroundStyle(color).overlay(Capsule().stroke(color.opacity(0.5), lineWidth: 1))
    }
}

struct BigButton: View {
    var title: LocalizedStringKey
    var icon: String = "arrow.up.right"
    var action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                BoltMark(color: BB.lime)
                Text(title).font(.headline).multilineTextAlignment(.leading)
                Spacer()
                Image(systemName: icon).font(.system(size: 19, weight: .bold))
            }
        }.buttonStyle(RouteButtonStyle())
    }
}

struct SectionLabel: View {
    var title: LocalizedStringKey
    var trailing: String = ""
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).font(.title3.bold())
            Spacer()
            if !trailing.isEmpty {
                Text(verbatim: trailing).font(.system(size: 10, weight: .medium, design: .monospaced)).tracking(1).foregroundStyle(BB.muted).lineLimit(1)
            }
        }
    }
}

struct Stat: View {
    var value: String
    var label: LocalizedStringKey
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(verbatim: value).font(BB.display(30)).monospacedDigit()
            Text(label).font(.caption).foregroundStyle(BB.muted)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

// Original vector climbing poster: crisp at every screen size, no remote asset dependencies.
struct ClimbingArt: View {
    var body: some View {
        GeometryReader { geometry in
            let w = geometry.size.width, h = geometry.size.height
            ZStack {
                Canvas { context, size in
                    for row in 0..<11 {
                        for column in 0..<11 {
                            let rect = CGRect(x: CGFloat(column)*size.width/10, y: CGFloat(row)*size.height/10, width: 2, height: 2)
                            context.fill(Path(ellipseIn: rect), with: .color(BB.ink.opacity(0.13)))
                        }
                    }
                    var route = Path()
                    route.move(to: CGPoint(x: w*0.18, y: h*0.93))
                    route.addCurve(to: CGPoint(x: w*0.80, y: h*0.08), control1: CGPoint(x: w*0.92,y:h*0.70), control2: CGPoint(x:w*0.15,y:h*0.45))
                    context.stroke(route, with: .color(BB.ink.opacity(0.23)), style: StrokeStyle(lineWidth: 1.3, dash: [4,6]))
                }
                hold(width: w*0.23, height: h*0.10, color: BB.lavender, angle: -22).position(x:w*0.77,y:h*0.17)
                hold(width: w*0.19, height: h*0.09, color: BB.ink, angle: 20).position(x:w*0.26,y:h*0.38)
                hold(width: w*0.16, height: h*0.08, color: BB.lavender, angle: -10).position(x:w*0.71,y:h*0.75)
                hold(width: w*0.20, height: h*0.07, color: BB.ink, angle: 8).position(x:w*0.30,y:h*0.91)
                Image(systemName: "figure.climbing").resizable().scaledToFit()
                    .foregroundStyle(BB.ink).frame(width:w*0.62,height:h*0.73)
                    .rotationEffect(.degrees(-12)).position(x:w*0.54,y:h*0.54)
                Text(verbatim: "↑").font(.system(size: 40,weight:.light)).rotationEffect(.degrees(20)).position(x:w*0.16,y:h*0.16)
                Text(verbatim: "GO AGAIN.").font(.system(size:10,weight:.black,design:.monospaced)).tracking(2)
                    .rotationEffect(.degrees(-90)).position(x:w*0.96,y:h*0.50)
            }
        }.accessibilityHidden(true)
    }
    private func hold(width: CGFloat, height: CGFloat, color: Color, angle: Double) -> some View {
        RoundedRectangle(cornerRadius: 12).fill(color).frame(width:width,height:height)
            .overlay(Circle().fill(BB.paper.opacity(0.6)).frame(width:4,height:4))
            .rotationEffect(.degrees(angle))
    }
}

struct PaperPage<Content: View>: View {
    var spacing: CGFloat = 24
    @ViewBuilder var content: Content
    var body: some View {
        ScrollView { VStack(alignment:.leading,spacing:spacing) { content }.padding(20).padding(.bottom,4) }.scrollDismissesKeyboard(.interactively)
            .background(BB.paper).foregroundStyle(BB.ink)
    }
}
