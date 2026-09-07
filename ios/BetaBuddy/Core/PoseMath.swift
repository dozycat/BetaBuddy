import Foundation

// Pure geometry, independent of inference/UI. Coordinates are top-left normalized.
enum PoseMath {
    static let connections: [(Int, Int)] = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(27,31),(28,30),(30,32),(28,32)]

    static func angle(_ frame: PoseFrame, _ a: Int, _ b: Int, _ c: Int, aspect: Double) -> Double? {
        guard aspect > 0, aspect.isFinite, [a,b,c].allSatisfy({ frame.joints.indices.contains($0) && frame.joints[$0].valid }) else { return nil }
        let p = frame.joints[a], q = frame.joints[b], r = frame.joints[c]
        let ux = (p.x-q.x)*aspect, uy = p.y-q.y, vx = (r.x-q.x)*aspect, vy = r.y-q.y
        let length = hypot(ux,uy)*hypot(vx,vy)
        guard length > 0.000001 else { return nil }
        return acos(max(-1, min(1, (ux*vx+uy*vy)/length))) * 180 / .pi
    }

    static func moments(frames: [PoseFrame], aspect: Double) -> [Moment] {
        var result: [Moment] = []
        var lastRaised = -10.0, lastFast = -10.0
        for (index, frame) in frames.enumerated() {
            guard frame.joints.count == 33, let hip = frame.hip else { continue }
            // A high heel alone cannot establish contact with a hold: explicitly a candidate.
            let heels = [frame.joints[29], frame.joints[30]].filter(\.valid)
            if heels.contains(where: { $0.y < hip.y - 0.04 }), frame.time - lastRaised >= 3 {
                result.append(Moment(time: frame.time, title: "高抬脚 · 候选片段", detail: "画面中脚跟高于髋部。回看是否为跟勾或高踩；模型无法确认岩点接触。"))
                lastRaised = frame.time
            }
            if index > 0, let previous = frames[index-1].hip {
                let dt = frame.time - frames[index-1].time
                if dt > 0, dt <= 0.3 {
                    let speed = hypot((hip.x-previous.x)*aspect, hip.y-previous.y)/dt
                    if speed > 0.45, frame.time-lastFast >= 3 {
                        result.append(Moment(time: frame.time, title: "快速移动 · 回看一下", detail: "髋部在画面中的移动较快。可能是动态动作、下落或镜头移动，请结合原片判断。"))
                        lastFast = frame.time
                    }
                }
            }
        }
        return Array(result.sorted { $0.time < $1.time }.prefix(24))
    }
}
