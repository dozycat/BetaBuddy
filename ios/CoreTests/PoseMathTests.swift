import XCTest
@testable import BetaBuddyCore

final class PoseMathTests: XCTestCase {
    func frame(_ time:Double = 0) -> PoseFrame {
        PoseFrame(time:time,joints:Array(repeating:Joint(x:0.5,y:0.5,confidence:1),count:33))
    }
    func testPartialBackgroundDetectionIsNotDrawn() {
        var pose = frame()
        pose.joints[23].y = -0.02
        let result = PoseAnalysis(frames:[pose],sampledFrames:1,duration:1,aspectRatio:1)
        XCTAssertFalse(pose.hasReliableTorso)
        XCTAssertNil(result.nearestFrame(to:0))
    }
    func testSkeletonIsNotCarriedAcrossDetectionGaps() {
        let result = PoseAnalysis(frames:[frame(),frame(1)],sampledFrames:10,duration:1,aspectRatio:1)
        XCTAssertNil(result.nearestFrame(to:0.5))
        XCTAssertEqual(result.nearestFrame(to:0.91)?.time,1)
        XCTAssertEqual(result.nearestFrame(to:0.05)?.time,0)
    }
    func testInvalidCoordinatesAreNotReliable() {
        var pose = frame(); pose.joints[23].x = .nan
        XCTAssertFalse(pose.hasReliableTorso)
        pose.joints[23].x = 1.1
        XCTAssertFalse(pose.hasReliableTorso)
        pose.joints[23].x = 0.5; pose.joints[23].confidence = 0.59
        XCTAssertFalse(pose.hasReliableTorso)
    }
    func testArchiveRoundTripPreservesPoseAndWorldLandmarks() throws {
        var pose = frame()
        pose.world = (0..<33).map { [rounded4(Double($0) / 7), -0.25, 0.123456, 0.9] }
        let original = PoseAnalysis(frames:[pose,frame(1)],sampledFrames:3,duration:0.7,aspectRatio:0.5625)
        XCTAssertTrue(original.hasWorldLandmarks)
        let bytes = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(PoseAnalysis.self,from:bytes)
        XCTAssertEqual(decoded.frames.first?.joints.count,33)
        XCTAssertEqual(decoded.frames.first?.world,pose.world)
        XCTAssertNil(decoded.frames.last?.world)
        XCTAssertEqual(decoded.sampledFrames,3)
        XCTAssertEqual(clockText(61),"01:01")
        XCTAssertEqual(clockText(.nan),"00:00")
    }
    // Stored landmarks keep four decimals; the encoder then writes them without a long tail.
    func testStoredNumbersAreShort() throws {
        XCTAssertEqual(rounded4(0.123456),0.1235)
        XCTAssertEqual(rounded4(-1.00004),-1)
        let text = String(decoding:try JSONEncoder().encode([rounded4(1.0 / 3),rounded4(2.0 / 3),rounded4(-0.07)]),as:UTF8.self)
        XCTAssertEqual(text,"[0.3333,0.6667,-0.07]")
    }
    // Records written before world landmarks were kept, with fields this build no longer uses.
    func testOlderPoseRecordsDecodeAndNeedPoseRecognitionAgain() throws {
        let joints = Array(repeating:#"{"x":0.5,"y":0.5,"confidence":1}"#,count:33).joined(separator:",")
        let legacy = #"{"frames":[{"time":0.1,"joints":["# + joints + #"]}],"sampledFrames":4,"duration":2,"aspectRatio":0.5,"processingSeconds":1,"moments":[{"id":"8F9619FF-8B86-D011-B42D-00C04FC964FF","time":1,"title":"t","detail":"d","kind":"fastMove"}]}"#
        let decoded = try JSONDecoder().decode(PoseAnalysis.self,from:Data(legacy.utf8))
        XCTAssertEqual(decoded.frames.count,1)
        XCTAssertEqual(decoded.sampledFrames,4)
        XCTAssertTrue(decoded.frames[0].hasReliableTorso)
        XCTAssertFalse(decoded.hasWorldLandmarks)
        XCTAssertFalse(PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1).hasWorldLandmarks)
        let rewritten = String(decoding:try JSONEncoder().encode(decoded),as:UTF8.self)
        XCTAssertFalse(rewritten.contains("moments"))
        XCTAssertFalse(rewritten.contains("processingSeconds"))
    }
}
