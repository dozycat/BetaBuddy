import XCTest
@testable import BetaBuddyCore

final class PoseMathTests: XCTestCase {
    func frame(_ time:Double = 0) -> PoseFrame {
        PoseFrame(time:time,joints:Array(repeating:Joint(x:0.5,y:0.5,confidence:1),count:33))
    }
    func testPartialBackgroundDetectionIsNotDrawn() {
        var pose = frame()
        pose.joints[23].y = -0.02
        let result = PoseAnalysis(frames:[pose],sampledFrames:1,duration:1,aspectRatio:1,processingSeconds:1,moments:[])
        XCTAssertFalse(pose.hasReliableTorso)
        XCTAssertNil(result.nearestFrame(to:0))
    }
    func testAnglesRespectVideoAspectRatio() {
        var pose = frame()
        pose.joints[11] = Joint(x:0.25,y:0.25,confidence:1)
        pose.joints[13] = Joint(x:0.5,y:0.5,confidence:1)
        pose.joints[15] = Joint(x:0.75,y:0.25,confidence:1)
        XCTAssertEqual(PoseMath.angle(pose,11,13,15,aspect:1)!,90,accuracy:0.001)
        XCTAssertEqual(PoseMath.angle(pose,11,13,15,aspect:2)!,126.8698976,accuracy:0.001)
    }
    func testUnknownAndDegenerateAnglesAreAbsent() {
        var pose = frame()
        XCTAssertNil(PoseMath.angle(pose,11,13,15,aspect:1))
        pose.joints[11].confidence = 0.1
        XCTAssertNil(PoseMath.angle(pose,11,13,15,aspect:1))
        XCTAssertNil(PoseMath.angle(pose,-1,13,15,aspect:1))
        XCTAssertNil(PoseMath.angle(pose,11,13,15,aspect:.nan))
    }
    func testCoverageCountsMissingDetections() {
        let result = PoseAnalysis(frames:[frame(),frame(1)],sampledFrames:10,duration:1,aspectRatio:1,processingSeconds:1,moments:[])
        XCTAssertEqual(result.coverage,0.2)
        XCTAssertNil(result.nearestFrame(to:0.5))
        XCTAssertEqual(result.nearestFrame(to:0.91)?.time,1)
        XCTAssertEqual(result.nearestFrame(to:0.05)?.time,0)
    }
    func testNoVelocityAcrossTrackingGaps() {
        var a = frame(), b = frame(1)
        a.joints[23].y = 0.9; a.joints[24].y = 0.9
        b.joints[23].y = 0.1; b.joints[24].y = 0.1
        a.joints[29].confidence = 0; a.joints[30].confidence = 0
        XCTAssertTrue(PoseMath.moments(frames:[a,b],aspect:1).isEmpty)
    }
    func testRaisedHeelCandidatesAreDebounced() {
        let frames = (0..<20).map { index -> PoseFrame in
            var pose = frame(Double(index)/10)
            pose.joints[29].y = 0.2
            return pose
        }
        let moments = PoseMath.moments(frames:frames,aspect:1)
        XCTAssertEqual(moments.count,1)
        XCTAssertTrue(moments[0].title.contains("候选"))
    }
    func testInvalidCoordinatesCannotProduceHip() {
        var pose = frame(); pose.joints[23].x = .nan
        XCTAssertNil(pose.hip)
        pose.joints[23].x = 1.1
        XCTAssertNil(pose.hip)
    }
    func testArchiveRoundTripPreservesAnalysis() throws {
        let original = PoseAnalysis(frames:[frame()],sampledFrames:3,duration:0.7,aspectRatio:0.5625,processingSeconds:2,moments:[])
        let decoded = try JSONDecoder().decode(PoseAnalysis.self,from:JSONEncoder().encode(original))
        XCTAssertEqual(decoded.frames.first?.joints.count,33)
        XCTAssertEqual(decoded.coverage,1.0/3,accuracy:0.0001)
        XCTAssertEqual(clockText(61),"01:01")
        XCTAssertEqual(clockText(.nan),"00:00")
    }
}
