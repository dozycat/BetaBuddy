import XCTest
@testable import BetaBuddyCore

/// The capture page's messages and the calls into it. Everything here is synthetic.
final class CaptureTests: XCTestCase {
    private let full: [String: Any] = [
        "type": "captureAssessment", "t": 12.4, "ok": false, "score": 0.62, "steady": true,
        "issues": [["code": "cut_top", "severity": "block", "move": "back"], ["code": "start_unclear", "severity": "warn"]],
        "hint": ["code": "cut_top", "move": "back"],
        "person": "none",
        "wall": ["top": 0.06, "bottom": 0.91, "left": 0, "right": 1],
        "palette": [["key": "yellow", "rgb": [226, 196, 40], "count": 9], ["key": "teal-ish", "rgb": [0, 300, -4], "count": 2]],
        "color": "yellow",
        "holds": [
            ["id": "h3", "box": [0.41, 0.33, 0.06, 0.04], "poly": [[0.41, 0.34], [0.44, 0.33], [0.47, 0.36]], "role": "start_hand", "conf": 0.8, "user": false],
            ["id": "h7", "box": [0.2, 0.9, 0.05, 0.03], "role": "start_foot", "user": true],
            ["id": "h12", "poly": [[0.5, 0.1], [0.6, 0.1], [0.6, 0.2], [0.5, 0.2]], "role": "finish"],
            ["id": "h20", "box": [0.7, 0.5, 0.05, 0.03], "role": "volume_edge"]
        ],
        "start": ["hands": ["h3"], "feet": ["h7"], "conf": 0.7],
        "finish": ["id": "h12", "conf": 0.6]
    ]

    private func assessment(_ body: [String: Any]) throws -> CaptureAssessment {
        guard case .assessment(let value) = try XCTUnwrap(CaptureMessage(body: body)) else { XCTFail("not an assessment"); throw CocoaError(.coderInvalidValue) }
        return value
    }

    func testAssessmentDecodesEveryDocumentedField() throws {
        let value = try assessment(full)
        XCTAssertEqual(value.time, 12.4)
        XCTAssertFalse(value.ok); XCTAssertTrue(value.steady); XCTAssertFalse(value.ready)
        XCTAssertEqual(value.score, 0.62)
        XCTAssertEqual(value.issues, [.init(code: "cut_top", blocking: true, move: "back"), .init(code: "start_unclear", blocking: false, move: nil)])
        XCTAssertEqual(value.hint, .init(code: "cut_top", blocking: true, move: "back"))
        XCTAssertEqual(value.person, "none")
        XCTAssertEqual(try XCTUnwrap(value.wall).minY, 0.06, accuracy: 1e-9); XCTAssertEqual(try XCTUnwrap(value.wall).maxY, 0.91, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(value.wall).width, 1, accuracy: 1e-9)
        XCTAssertEqual(value.palette.map(\.key), ["yellow", "teal-ish"])
        XCTAssertEqual(value.palette[0].rgb, [226, 196, 40]); XCTAssertEqual(value.palette[0].count, 9)
        XCTAssertEqual(value.palette[1].rgb, [0, 255, 0])          // Out-of-range channels are clamped.
        XCTAssertEqual(value.color, "yellow")
        XCTAssertEqual(value.holds.map(\.id), ["h3", "h7", "h12", "h20"])
        XCTAssertEqual(value.holds[0].role, .startHand); XCTAssertEqual(value.holds[0].poly.count, 3); XCTAssertEqual(value.holds[0].confidence, 0.8); XCTAssertFalse(value.holds[0].user)
        XCTAssertEqual(value.holds[1].role, .startFoot); XCTAssertTrue(value.holds[1].poly.isEmpty); XCTAssertNil(value.holds[1].confidence); XCTAssertTrue(value.holds[1].user)
        // No box: it is taken from the outline.
        XCTAssertEqual(value.holds[2].role, .finish)
        XCTAssertEqual(value.holds[2].box.minX, 0.5, accuracy: 1e-9); XCTAssertEqual(value.holds[2].box.width, 0.1, accuracy: 1e-9)
        XCTAssertEqual(value.holds[3].role, .hold)                 // A role this build does not know is still a hold of the problem.
        XCTAssertEqual(value.startHands, ["h3"]); XCTAssertEqual(value.startFeet, ["h7"]); XCTAssertEqual(value.finish, "h12")
    }

    func testARecognisedProblemAndWhoMarkedTheStart() throws {
        var body = full
        XCTAssertNil(try assessment(body).route)
        XCTAssertFalse(try assessment(body).startConfirmed)        // h3 is the page's own guess…
        XCTAssertFalse(try assessment(body).finishConfirmed)
        body["route"] = ["id": "wall-07", "conf": 0.9, "name": "Synthetic 7", "nameZh": "合成线路 7", "grade": "V3", "colour": "yellow", "wall": "east",
                         "others": ["wall-08"], "otherRoutes": [["id": "wall-08", "name": "Synthetic 8", "grade": ""]], "extra": true]
        body["holds"] = [["id": "h3", "box": [0.4, 0.3, 0.1, 0.1], "role": "start_hand", "user": true], ["id": "h9", "box": [0.4, 0.1, 0.1, 0.1], "role": "finish", "user": true]]
        let value = try assessment(body)
        XCTAssertEqual(value.route, .init(id: "wall-07", confidence: 0.9, name: "Synthetic 7", nameZh: "合成线路 7", grade: "V3"))
        XCTAssertEqual(value.route?.title, "Synthetic 7"); XCTAssertEqual(value.route?.albumGrade, "V3")
        // What it is called without a name in the interface language, and without any name.
        XCTAssertEqual(CaptureAssessment.Route(id: "wall-07", nameZh: "合成线路 7").title, "合成线路 7")
        XCTAssertEqual(CaptureAssessment.Route(id: "wall-07").title, "wall-07")
        // Only a grade the album itself uses is ever filled in; gyms write grades in many other ways.
        for (text, expected) in [("V0", "V0"), ("V12", "V12"), ("V16", "V16"), ("V17", nil), ("v3", nil), ("V03", nil), ("V3+", nil), ("紫条 11–12", nil), ("", nil)] as [(String, String?)] {
            XCTAssertEqual(CaptureAssessment.Route(id: "r", grade: text).albumGrade, expected, text)
        }
        XCTAssertNil(CaptureAssessment.Route(id: "r").albumGrade)
        XCTAssertEqual(CaptureAssessment.Route(body: ["id": "r", "grade": ""])?.grade, nil)
        XCTAssertTrue(value.startConfirmed); XCTAssertTrue(value.finishConfirmed)     // …until the user taps.
        // Only an id is promised; without one there is no problem.
        body["route"] = ["id": "wall-07"]
        XCTAssertEqual(try assessment(body).route, .init(id: "wall-07"))
        for junk in [["conf": 0.9], ["id": ""], "wall-07", NSNull()] as [Any] { body["route"] = junk; XCTAssertNil(try assessment(body).route) }
        // The same in the summary that comes with the plan.
        guard case .locked(_, _, let summary) = try XCTUnwrap(CaptureMessage(body: ["type": "captureLocked", "id": "A", "plan": "p", "summary": ["holds": 3, "route": ["id": "wall-07", "conf": 0.8]]])) else { return XCTFail() }
        XCTAssertEqual(summary.route, .init(id: "wall-07", confidence: 0.8)); XCTAssertEqual(summary.holds, 3)
    }

    func testAssessmentWithOnlyItsTypeIsReadCautiously() throws {
        let value = try assessment(["type": "captureAssessment"])
        XCTAssertTrue(value.ok)            // No issue blocks.
        XCTAssertFalse(value.steady)       // Nothing said the picture is still.
        XCTAssertFalse(value.ready)
        XCTAssertNil(value.score); XCTAssertNil(value.hint); XCTAssertNil(value.wall); XCTAssertNil(value.color); XCTAssertNil(value.finish)
        XCTAssertTrue(value.issues.isEmpty); XCTAssertTrue(value.palette.isEmpty); XCTAssertTrue(value.holds.isEmpty)
        XCTAssertEqual(value.person, "unknown")
    }

    func testMissingAndNullOptionalFieldsAndUnknownCodes() throws {
        var body = full
        body["hint"] = NSNull(); body["wall"] = NSNull(); body["color"] = NSNull(); body["finish"] = NSNull()
        body["issues"] = [["code": "lens_flare_2031", "severity": "block", "move": "spin_around"], ["code": "soft_light", "severity": "nudge"], ["severity": "block"], "junk"]
        body["start"] = ["hands": [], "feet": []]
        body.removeValue(forKey: "ok"); body.removeValue(forKey: "score"); body.removeValue(forKey: "steady")
        let value = try assessment(body)
        // Unknown codes and moves are kept as they came; a severity that is not “warn” blocks.
        XCTAssertEqual(value.issues, [.init(code: "lens_flare_2031", blocking: true, move: "spin_around"), .init(code: "soft_light", blocking: true, move: nil)])
        XCTAssertFalse(value.ok)                                    // Derived from the issues when the page does not say.
        XCTAssertEqual(value.hint?.code, "lens_flare_2031")         // Without a hint the first issue is the one to show.
        XCTAssertNil(value.wall); XCTAssertNil(value.color); XCTAssertNil(value.finish)
        XCTAssertTrue(value.startHands.isEmpty)
        XCTAssertNil(CaptureGuidance.Code(rawValue: "lens_flare_2031"))
        XCTAssertNil(CaptureGuidance.Move(rawValue: "spin_around"))
    }

    func testWrongTypesNeverCrashAndBadPartsAreDropped() throws {
        let value = try assessment([
            "type": "captureAssessment", "t": "soon", "ok": "yes", "score": 7, "steady": 1, "issues": "none", "hint": 3,
            "person": 5, "wall": ["top": 0.9, "bottom": 0.1, "left": 0, "right": 1], "palette": [["rgb": [1, 2, 3]], ["key": "blue", "rgb": "blue"], 4],
            "holds": [["id": "a"], ["box": [0, 0, 1, 1]], ["id": "b", "box": [0.1, 0.1, "wide", 0.1]], ["id": "c", "box": [0.1, 0.2, 0.3, 0.4], "poly": [[0.1], "x"], "role": "none"]],
            "start": ["hands": "h1", "feet": [1, "f1", ""]], "finish": "top"
        ])
        XCTAssertEqual(value.time, 0); XCTAssertTrue(value.ok); XCTAssertEqual(value.score, 1); XCTAssertFalse(value.steady)
        XCTAssertNil(value.wall)                                     // Upside-down wall.
        XCTAssertEqual(value.palette.map(\.key), ["blue"]); XCTAssertEqual(value.palette[0].rgb, [128, 128, 128])
        XCTAssertEqual(value.holds.map(\.id), ["c"])                 // Only the hold that can be drawn.
        XCTAssertTrue(value.holds[0].poly.isEmpty); XCTAssertEqual(value.holds[0].role, .hold)
        XCTAssertTrue(value.startHands.isEmpty); XCTAssertEqual(value.startFeet, ["f1"]); XCTAssertNil(value.finish)
        let many = (0..<1000).map { ["id": "h\($0)", "box": [0, 0, 0.1, 0.1]] as [String: Any] }
        XCTAssertEqual(try assessment(["type": "captureAssessment", "holds": many]).holds.count, CaptureAssessment.maximumHolds)
    }

    func testOtherMessages() throws {
        XCTAssertEqual(CaptureMessage(body: ["type": "captureReady", "packVersion": "1.9.0"]), .ready(packVersion: "1.9.0"))
        XCTAssertEqual(CaptureMessage(body: ["type": "captureReady"]), .ready(packVersion: ""))
        XCTAssertEqual(CaptureMessage(body: ["type": "captureError", "code": "no_canvas", "message": "x"]), .failed(code: "no_canvas", message: "x"))
        XCTAssertEqual(CaptureMessage(body: ["type": "error", "message": "boom"]), .pageError("boom"))
        for other in [["type": "analysisReady"], ["type": "ready"], ["type": "captureProgress"], ["kind": "captureReady"]] as [[String: Any]] {
            XCTAssertNil(CaptureMessage(body: other))
        }
        XCTAssertNil(CaptureMessage(body: "captureReady")); XCTAssertNil(CaptureMessage(body: 3))

        let summary: [String: Any] = ["color": "yellow", "rgb": [226, 196, 40], "holds": 9, "start": ["hands": 2, "feet": 1], "finish": true, "ok": false, "issues": ["cut_top", "new_code"]]
        let expected = CaptureSummary(color: "yellow", rgb: [226, 196, 40], holds: 9, startHands: 2, startFeet: 1, finish: true, ok: false, issues: ["cut_top", "new_code"])
        XCTAssertEqual(CaptureMessage(body: ["type": "captureLocked", "id": "A1", "plan": "{\"v\":1}", "summary": summary]), .locked(id: "A1", plan: "{\"v\":1}", summary: expected))
        // A plan handed over as an object is kept as its JSON text; an empty or oversized one is not kept.
        guard case .locked(_, let asObject, _) = try XCTUnwrap(CaptureMessage(body: ["type": "captureLocked", "id": "A1", "plan": ["v": 1]])) else { return XCTFail() }
        XCTAssertEqual(asObject, "{\"v\":1}")
        XCTAssertEqual(CaptureMessage(body: ["type": "captureLocked", "id": "A1", "plan": ""]), .locked(id: "A1", plan: nil, summary: CaptureSummary()))
        XCTAssertEqual(CaptureMessage(body: ["type": "captureLocked", "plan": String(repeating: "x", count: CaptureMessage.maximumPlanLength + 1), "summary": ["color": NSNull()]]),
                       .locked(id: "", plan: nil, summary: CaptureSummary()))
    }

    func testCommandsBecomeJavaScriptLiterals() throws {
        let begin = CaptureCommand.begin(id: "7F96", width: 360, height: 640, fieldOfView: 38.512345, venue: "synthetic-east").script
        XCTAssertTrue(begin.hasPrefix("window.BetaBuddyCapture && typeof window.BetaBuddyCapture.begin === 'function' ? (window.BetaBuddyCapture.begin("))
        for part in ["\"id\":\"7F96\"", "\"w\":360", "\"h\":640", "\"fovH\":38.51", "\"venue\":\"synthetic-east\""] { XCTAssertTrue(begin.contains(part), part) }
        let bare = CaptureCommand.begin(id: "a", width: 1, height: 2, fieldOfView: nil, venue: nil).script
        XCTAssertFalse(bare.contains("fovH")); XCTAssertFalse(bare.contains("venue"))
        XCTAssertFalse(CaptureCommand.begin(id: "a", width: 1, height: 2, fieldOfView: .nan, venue: "").script.contains("fovH"))
        // Ids cannot break out of the call.
        XCTAssertTrue(CaptureCommand.begin(id: "\");alert(1);(\"", width: 1, height: 1, fieldOfView: nil, venue: nil).script.contains("\\\");alert(1);(\\\""))

        let plain = CaptureCommand.addFrame(time: 1.23456, jpegBase64: "/9j/AA==", info: CaptureFrameInfo()).script
        XCTAssertTrue(plain.contains("addFrame(1.235,\"/9j/AA==\")"), plain)       // No third argument when there is nothing to say.
        let pose = [[Double]](repeating: [0.123456, 0.5, 0.98765], count: 33)
        let rich = CaptureCommand.addFrame(time: 2, jpegBase64: "AA", info: CaptureFrameInfo(pitch: 12.3456, roll: -1.004, shake: 0.012345, pose: pose)).script
        for part in ["\"pitch\":12.35", "\"roll\":-1", "\"shake\":0.0123", "\"pose\":[[0.1235,0.5,0.9877]"] { XCTAssertTrue(rich.contains(part), part) }
        let partial = CaptureCommand.addFrame(time: 2, jpegBase64: "AA", info: CaptureFrameInfo(pitch: 3)).script
        XCTAssertTrue(partial.contains("{\"pitch\":3}"), partial)

        XCTAssertTrue(CaptureCommand.setColor("yellow").script.contains("setColor(\"yellow\")"))
        XCTAssertTrue(CaptureCommand.setColor(nil).script.contains("setColor(null)"))
        XCTAssertTrue(CaptureCommand.tap(x: 0.41234567, y: 1.7, role: .startFoot).script.contains("(window.BetaBuddyCapture.tap(0.4123,1,\"start_foot\") || '')"))
        XCTAssertTrue(CaptureCommand.tap(x: 0.5, y: 0.5, role: .none).script.contains("\"none\""))
        XCTAssertTrue(CaptureCommand.resetTaps.script.contains("resetTaps()"))
        let reject = CaptureCommand.rejectRoute("venue/wall-07\"x").script
        XCTAssertTrue(reject.contains("typeof window.BetaBuddyCapture.rejectRoute === 'function'")); XCTAssertTrue(reject.contains("rejectRoute(\"venue/wall-07\\\"x\"), true)"), reject)
        XCTAssertTrue(CaptureCommand.lock.script.contains("lock()"))
        XCTAssertEqual(Set(CaptureRole.allCases.map(\.rawValue)), ["start_hand", "start_foot", "finish", "hold", "none"])
    }

    func testEveryCodeAndMoveOfTheInterfaceIsKnown() {
        let codes = ["no_wall", "too_dark", "backlit", "blurry", "shaking", "tilted", "too_close", "too_far", "cut_top", "cut_bottom", "cut_left", "cut_right",
                     "oblique", "pitch_high", "occluded", "no_holds", "start_unclear", "finish_unclear"]
        XCTAssertEqual(Set(CaptureGuidance.Code.allCases.map(\.rawValue)), Set(codes))
        let moves = ["back", "closer", "left", "right", "up", "down", "tilt_up", "tilt_down", "level", "hold_still"]
        XCTAssertEqual(Set(CaptureGuidance.Move.allCases.map(\.rawValue)), Set(moves))
    }

    func testPhoneAnglesFromGravity() throws {
        // Upright portrait: camera level, no roll.
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.pitch(gravity: (0, -1, 0))), 0, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.roll(gravity: (0, -1, 0))), 0, accuracy: 1e-9)
        // Leaning so the camera looks up by 30°.
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.pitch(gravity: (0, -cos(.pi / 6), sin(.pi / 6)))), 30, accuracy: 1e-9)
        // Screen up on a table: the camera looks straight down, and there is no roll to speak of.
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.pitch(gravity: (0, 0, -1))), -90, accuracy: 1e-9)
        XCTAssertEqual(CaptureMotion.roll(gravity: (0, 0, -1)), 0)
        XCTAssertNil(CaptureMotion.roll(gravity: (0, .nan, 0)))
        // Turned clockwise as seen on the screen: its right edge points down.
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.roll(gravity: (sin(.pi / 18), -cos(.pi / 18), 0))), 10, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.roll(gravity: (1, 0, 0))), 90, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.roll(gravity: (-sin(.pi / 18), -cos(.pi / 18), 0))), -10, accuracy: 1e-9)
        XCTAssertNil(CaptureMotion.pitch(gravity: (.nan, 0, 0)))
        // 16:9 sensor with 70° across its long side, held upright.
        XCTAssertEqual(try XCTUnwrap(CaptureMotion.uprightFieldOfView(sensorDegrees: 70, longSide: 1920, shortSide: 1080)), 43.0, accuracy: 0.1)
        XCTAssertNil(CaptureMotion.uprightFieldOfView(sensorDegrees: 0, longSide: 1920, shortSide: 1080))
    }

    private func climb() -> Climb {
        Climb(id: UUID(), date: Date(timeIntervalSince1970: 1000), title: "", grade: "V3", sent: false, note: "keep", videoName: "v.mov", thumbnailName: "t.jpg",
              analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 0.5625))
    }

    func testRecordsFromBeforeTheCameraStillLoad() throws {
        // What 1.6.0 wrote: no `capture` at all.
        let old = """
        {"version":1,"folders":[],"climbs":[{"id":"7F9619FF-8B86-D011-B42D-00C04FC964FF","date":700000000,"title":"old","grade":"V2","sent":true,"note":"n",
          "videoName":"7F9619FF-8B86-D011-B42D-00C04FC964FF.mov","thumbnailName":"7F9619FF-8B86-D011-B42D-00C04FC964FF.jpg","gymId":"g","gymName":"Gym","climbedAt":700000100,
          "analysis":{"frames":[],"sampledFrames":0,"duration":12.5,"aspectRatio":0.5625,"metrics":{"gone":1}}}]}
        """
        let archive = try JSONDecoder().decode(Archive.self, from: Data(old.utf8))
        XCTAssertEqual(archive.climbs.count, 1)
        XCTAssertNil(archive.climbs[0].capture)
        XCTAssertEqual(archive.climbs[0].title, "old"); XCTAssertEqual(archive.climbs[0].gymKey, "id:g"); XCTAssertNil(archive.climbs[0].pendingAnalysis)
        // Written again and read again, it is still an imported record.
        let again = try JSONDecoder().decode(Archive.self, from: JSONEncoder().encode(archive))
        XCTAssertNil(again.climbs[0].capture)
    }

    func testCameraRecordRoundTripsAndToleratesPartialCaptureInfo() throws {
        var record = climb()
        record.capture = CaptureInfo(colorKey: "yellow", colorRGB: [226, 196, 40],
                                     summary: CaptureSummary(color: "yellow", rgb: [226, 196, 40], holds: 9, startHands: 2, startFeet: 1, finish: true, ok: true, issues: []),
                                     hasPlan: true, packVersion: "1.9.0", analysisQueued: true, routeId: "wall-07", routeName: "合成线路 7",
                                     planUse: CapturePlanUse(used: false, reason: "mismatch"))
        let bytes = try JSONEncoder().encode(Archive(climbs: [record]))
        XCTAssertEqual(try JSONDecoder().decode(Archive.self, from: bytes).climbs[0].capture, record.capture)
        // A later build may add fields and leave optional ones out.
        var root = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        var climbs = try XCTUnwrap(root["climbs"] as? [[String: Any]])
        climbs[0]["capture"] = ["hasPlan": false, "somethingNew": [1, 2, 3]]
        root["climbs"] = climbs
        let sparse = try JSONDecoder().decode(Archive.self, from: JSONSerialization.data(withJSONObject: root)).climbs[0].capture
        XCTAssertEqual(sparse, CaptureInfo())
    }
}
