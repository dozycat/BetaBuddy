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

struct Stamp: View {
    var text: String
    var color: Color = BB.ink
    var body: some View {
        Text(text).font(.system(size: 10, weight: .bold, design: .monospaced)).tracking(1.3).lineLimit(1).minimumScaleFactor(0.7)
            .padding(.horizontal, 9).padding(.vertical, 6)
            .foregroundStyle(color).overlay(Capsule().stroke(color.opacity(0.5), lineWidth: 1))
    }
}

struct BigButton: View {
    var title: String
    var icon: String = "arrow.up.right"
    var action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                Image(systemName: icon).font(.system(size: 19, weight: .bold))
            }.padding(20).foregroundStyle(BB.paper).background(BB.ink, in: RoundedRectangle(cornerRadius: 20))
        }.buttonStyle(.plain)
    }
}

struct SectionLabel: View {
    var title: String
    var trailing: String
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).font(.title3.bold())
            Spacer()
            Text(trailing).font(.system(size: 10, weight: .medium, design: .monospaced)).tracking(1).foregroundStyle(BB.muted)
        }
    }
}

struct Stat: View {
    var value: String
    var label: String
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(value).font(BB.display(30)).monospacedDigit()
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
                Text("↑").font(.system(size: 40,weight:.light)).rotationEffect(.degrees(20)).position(x:w*0.16,y:h*0.16)
                Text("GO AGAIN.").font(.system(size:10,weight:.black,design:.monospaced)).tracking(2)
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
    @ViewBuilder var content: Content
    var body: some View {
        ScrollView { VStack(alignment:.leading,spacing:28) { content }.padding(24).padding(.bottom,24) }.scrollDismissesKeyboard(.interactively)
            .background(BB.paper).foregroundStyle(BB.ink)
    }
}
