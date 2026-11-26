import XCTest
@testable import BetaBuddyCore

final class EditPlanTests: XCTestCase {
    /// A made-up plan in the form the analysis page sends. Nothing in it comes from a real clip.
    static let json = """
    {"v":1,"duration":10,"trim":[1,9],"unknown":{"later":true},
     "segments":[{"t0":1,"t1":4,"speed":1,"kind":"run"},{"t0":4,"t1":5,"speed":0.5,"kind":"slow"},{"t0":5,"t1":9,"speed":2,"kind":"fast"},
                 {"t0":3.6,"t1":5.2,"speed":0.25,"kind":"replay","camera":[{"t":3.6,"cx":0.5,"cy":0.4,"s":2},{"t":5.2,"cx":0.6,"cy":0.3,"s":3}]}],
     "camera":[{"t":6,"cx":0.2,"cy":0.9,"s":2},{"t":1,"cx":0.5,"cy":0.5,"s":1},{"t":"soon","cx":0,"cy":0,"s":1}],
     "keep":{"feather":0.03,"boxes":[{"t":2,"b":[[0.3,0.4,0.2,0.4]]},{"t":1,"b":[[0.1,0.4,0.2,0.4]]},{"t":3,"b":[[0.3,0.4,0.2,0.4],[0.7,0.1,0.1,0.1]]}]},
     "plate":{"times":[0.5,2,3.5,5,6.5,8,9.5,11,-3]},
     "marks":[{"t0":4,"t1":5,"kind":"dyno"}]}
    """

    func testPlanIsReadSortedAndUnknownPartsAreIgnored() throws {
        let plan = try XCTUnwrap(EditPlan(json: Self.json))
        XCTAssertEqual(plan.duration, 10)
        XCTAssertEqual(plan.trim, 1...9)
        XCTAssertEqual(plan.segments.map(\.speed), [1, 0.5, 2, 0.25])
        XCTAssertNil(plan.segments[0].camera)
        XCTAssertEqual(plan.segments[3].camera?.count, 2)
        XCTAssertEqual(plan.camera.map(\.t), [1, 6])                       // Sorted; the unreadable keyframe is left out.
        XCTAssertEqual(plan.keep?.samples.map(\.t), [1, 2, 3])
        XCTAssertEqual(plan.plateTimes, [0.5, 2, 3.5, 5, 6.5, 8, 9.5])     // Times outside the clip are left out.
        XCTAssertTrue(plan.offersRemoval); XCTAssertTrue(plan.offersSlowMotion); XCTAssertTrue(plan.offersCamera)
    }

    func testPlansThisBuildCannotRenderAreRefused() throws {
        func plan(_ edit: (inout [String: Any]) -> Void) throws -> EditPlan? {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(Self.json.utf8)) as? [String: Any])
            edit(&object)
            return EditPlan(object: object)
        }
        XCTAssertNotNil(try plan { _ in })
        XCTAssertNil(try plan { $0["v"] = 2 })
        XCTAssertNil(try plan { $0["v"] = true })
        XCTAssertNil(try plan { $0["duration"] = 0 })
        XCTAssertNil(try plan { $0["segments"] = [] as [Any] })
        XCTAssertNil(try plan { $0["segments"] = Array(repeating: ["t0": 0, "t1": 1, "speed": 1], count: 41) })
        XCTAssertNil(try plan { $0["segments"] = [["t0": 0, "t1": 1, "speed": 5]] })
        XCTAssertNil(try plan { $0["segments"] = [["t0": 0, "t1": 1, "speed": 0.1]] })
        XCTAssertNil(try plan { $0["segments"] = [["t0": 2, "t1": 1, "speed": 1]] })
        XCTAssertNil(try plan { $0["segments"] = [["t0": 0, "t1": 0.1, "speed": 1]] })
        XCTAssertNil(try plan { $0["segments"] = [["t0": 0, "t1": 12, "speed": 1]] })
        XCTAssertNil(EditPlan(json: "not json")); XCTAssertNil(EditPlan(object: nil)); XCTAssertNil(EditPlan(object: [1, 2]))
        // Optional parts that are missing or malformed only take their own switch away.
        let bare = try XCTUnwrap(try plan { $0["keep"] = nil; $0["camera"] = "wide"; $0["trim"] = nil; $0["segments"] = [["t0": 2, "t1": 6, "speed": 1]] })
        XCTAssertNil(bare.keep); XCTAssertTrue(bare.plateTimes.isEmpty); XCTAssertTrue(bare.camera.isEmpty)
        XCTAssertEqual(bare.trim, 2...6)
        XCTAssertFalse(bare.offersRemoval); XCTAssertFalse(bare.offersSlowMotion); XCTAssertFalse(bare.offersCamera)
        XCTAssertNil(try XCTUnwrap(try plan { $0["plate"] = nil }).keep)   // Nothing to put in a passer-by's place.
        // Out-of-range camera values are brought into range.
        let wide = try XCTUnwrap(try plan { $0["camera"] = [["t": 0, "cx": -1, "cy": 2, "s": 9]] })
        XCTAssertEqual(wide.camera, [EditPlan.CameraKey(t: 0, cx: 0, cy: 1, s: 8)])   // The format allows up to 8.
    }

    func testOddPlansDoNotBringTheAppDown() throws {
        func plan(_ edit: (inout [String: Any]) -> Void) throws -> EditPlan? {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(Self.json.utf8)) as? [String: Any])
            edit(&object)
            return EditPlan(object: object)
        }
        // A trim that is only inside the clip by the rounding allowance has nothing left once cut to the clip:
        // the segments say what the attempt is instead.
        for odd in [[10.01, 10.04], [-0.04, -0.01], [9.99, 10.03], [5, 5], [6, 5]] {
            let read = try XCTUnwrap(try plan { $0["trim"] = odd }, "\(odd)")
            XCTAssertEqual(read.trim, 1...9, "\(odd)")
        }
        XCTAssertEqual(try XCTUnwrap(try plan { $0["trim"] = [-0.04, 4] }).trim, 0...4)          // Cut to the clip, most of it left.
        // Boxes are cut to the picture; one that lies outside it, or has no size, is no box. Wild numbers included.
        let boxes: [[Double]] = [[1e30, 0.1, 0.2, 0.2], [-1e30, 0.1, 0.2, 0.2], [0.1, 0.1, 1e300, 1e300], [-0.2, 0.9, 0.5, 0.5], [0.2, 0.2, 0, 0.1], [0.2, 0.2, 0.1, -1], [1.2, 0.2, 0.1, 0.1]]
        let wild = try XCTUnwrap(try plan { $0["keep"] = ["feather": 5, "boxes": [["t": 1, "b": boxes]]] })
        let kept = try XCTUnwrap(wild.keep)
        XCTAssertEqual(kept.feather, 0.2)
        XCTAssertEqual(kept.samples.first?.boxes.count, 2)
        XCTAssertEqual(kept.samples.first?.boxes.first, EditPlan.Box(x: 0.1, y: 0.1, w: 0.9, h: 0.9))
        let corner = try XCTUnwrap(kept.samples.first?.boxes.last)
        XCTAssertEqual(corner.x, 0); XCTAssertEqual(corner.y, 0.9); XCTAssertEqual(corner.w, 0.3, accuracy: 1e-9); XCTAssertEqual(corner.h, 0.1, accuracy: 1e-9)
        // Turning a box into rows and columns never leaves the picture, whatever the numbers.
        XCTAssertEqual(EditPlan.Box.cells(0.25, 0.5, of: 100), 25..<75)
        XCTAssertEqual(EditPlan.Box.cells(-3, 3.5, of: 100), 0..<50)
        XCTAssertEqual(EditPlan.Box.cells(0.9, 7, of: 100), 90..<100)
        XCTAssertEqual(EditPlan.Box.cells(1e30, 0.2, of: 100), 100..<100)
        XCTAssertEqual(EditPlan.Box.cells(-1e30, 1e30, of: 100), 0..<0)
        XCTAssertEqual(EditPlan.Box.cells(.nan, .infinity, of: 100), 0..<0)
        XCTAssertTrue(EditPlan.Box.cells(0.5, -0.2, of: 100).isEmpty)
        // A segment whose own camera is null (or not a list) has none of its own: the plan's camera applies.
        let nulled = try XCTUnwrap(try plan { $0["segments"] = [["t0": 1, "t1": 4, "speed": 1, "camera": NSNull()], ["t0": 4, "t1": 6, "speed": 1, "camera": "close"], ["t0": 6, "t1": 8, "speed": 1, "camera": [Any]()]] })
        XCTAssertNil(nulled.segments[0].camera); XCTAssertNil(nulled.segments[1].camera)
        XCTAssertEqual(nulled.segments[2].camera, [])                                             // An empty list is its own: the full picture.
        let timeline = EditTimeline(plan: nulled, options: EditOptions(), sourceDuration: 10)
        XCTAssertEqual(timeline.pieces.map(\.camera.count), [2, 2, 0])
    }

    func testTimelineMapsTheResultBackToTheClip() throws {
        let plan = try XCTUnwrap(EditPlan(json: Self.json))
        let timeline = EditTimeline(plan: plan, options: EditOptions(), sourceDuration: 10)
        XCTAssertEqual(timeline.pieces.map(\.start), [0, 3, 5, 7])
        XCTAssertEqual(timeline.duration, 3 + 2 + 2 + 1.6 / 0.25, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(timeline.source(atOutput: 0)).time, 1)
        XCTAssertEqual(try XCTUnwrap(timeline.source(atOutput: 4)).time, 4.5, accuracy: 1e-9)       // Half speed.
        XCTAssertEqual(try XCTUnwrap(timeline.source(atOutput: 6)).time, 7, accuracy: 1e-9)         // Double speed.
        let replay = try XCTUnwrap(timeline.source(atOutput: 7 + 3.2))
        XCTAssertEqual(replay.piece, 3); XCTAssertEqual(replay.time, 4.4, accuracy: 1e-9)           // Back in time, a quarter speed.
        XCTAssertEqual(try XCTUnwrap(timeline.source(atOutput: 99)).time, 5.2)                      // Past the end: the last moment.
        XCTAssertEqual(timeline.pieces[3].camera.count, 2)                                          // Its own camera.
        XCTAssertEqual(timeline.pieces[0].camera.map(\.t), [1, 6])                                  // The plan's.

        // Slow motion off: the attempt alone, at its own speed. Camera off: no keyframes anywhere.
        let plain = EditTimeline(plan: plan, options: EditOptions(removePassersBy: false, slowMotion: false, cameraMoves: false), sourceDuration: 10)
        XCTAssertEqual(plain.pieces, [EditTimeline.Piece(t0: 1, t1: 9, speed: 1, start: 0, camera: [])])
        XCTAssertEqual(plain.duration, 8)

        // A clip shorter than the plan thought: ranges are cut to it and what is left of nothing is dropped.
        let short = EditTimeline(plan: plan, options: EditOptions(), sourceDuration: 4.02)
        XCTAssertEqual(short.pieces.map(\.t1), [4, 4.02])
        XCTAssertEqual(short.pieces.map(\.t0), [1, 3.6])
        XCTAssertTrue(EditTimeline(plan: plan, options: EditOptions(), sourceDuration: 0.5).pieces.isEmpty)
        XCTAssertNil(EditTimeline(plan: plan, options: EditOptions(), sourceDuration: 0.5).source(atOutput: 0))
    }

    func testAPlanThatTracksItsSubjectIsFollowedInStraightLines() throws {
        let tracking = try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":10,"cameraMode":"track","segments":[{"t0":0,"t1":10,"speed":1}],"camera":[{"t":0,"cx":0.5,"cy":0.5,"s":1},{"t":4,"cx":0.5,"cy":0.5,"s":3}]}"#))
        XCTAssertTrue(tracking.tracksSubject)
        let piece = try XCTUnwrap(EditTimeline(plan: tracking, options: EditOptions(), sourceDuration: 10).pieces.first)
        XCTAssertTrue(piece.linear)
        XCTAssertEqual(EditWindow.at(1, keys: piece.camera, linear: piece.linear).s, 1.5, accuracy: 1e-9)
        // Without the field, or with a value this build does not know, moves are eased as before.
        let eased = try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":10,"cameraMode":"later","segments":[{"t0":0,"t1":10,"speed":1}]}"#))
        XCTAssertFalse(eased.tracksSubject)
        XCTAssertFalse(try XCTUnwrap(EditTimeline(plan: eased, options: EditOptions(), sourceDuration: 10).pieces.first).linear)
    }

    func testCameraEasesBetweenKeyframesAndStaysInsideThePicture() {
        let keys = [EditPlan.CameraKey(t: 1, cx: 0.5, cy: 0.5, s: 1), EditPlan.CameraKey(t: 3, cx: 0.5, cy: 0.5, s: 3)]
        XCTAssertEqual(EditWindow.at(0, keys: keys), .full)
        XCTAssertEqual(EditWindow.at(9, keys: keys).s, 3)
        XCTAssertEqual(EditWindow.at(2, keys: keys).s, 2, accuracy: 1e-9)             // Halfway through a smoothstep is halfway.
        XCTAssertEqual(EditWindow.at(1.5, keys: keys).s, 1 + 2 * 0.15625, accuracy: 1e-9)   // Slower at the ends: 3u² − 2u³ at u = ¼.
        XCTAssertEqual(EditWindow.at(5, keys: []), .full)
        // A dense track of the subject is followed in straight lines: a quarter of the way is a quarter of the move.
        XCTAssertEqual(EditWindow.at(1.5, keys: keys, linear: true).s, 1.5, accuracy: 1e-9)
        XCTAssertEqual(EditWindow.at(9, keys: keys, linear: true).s, 3)
        // A window that would hang over an edge is moved in.
        let corner = EditWindow.at(0, keys: [EditPlan.CameraKey(t: 0, cx: 0.05, cy: 0.98, s: 2)])
        XCTAssertEqual(corner.cx, 0.25); XCTAssertEqual(corner.cy, 0.75)
        XCTAssertEqual(corner.x, 0); XCTAssertEqual(corner.y, 0.5); XCTAssertEqual(corner.side, 0.5)
        XCTAssertEqual(EditWindow(cx: .nan, cy: 0.5, s: 70).clamped, EditWindow(cx: 0.5, cy: 0.5, s: 8))
        XCTAssertEqual(EditWindow(cx: 0.9, cy: 0.1, s: 0.5).clamped, .full)
    }

    func testKeepBoxesMoveBoxByBoxAndOtherwiseTakeTheNearerSample() throws {
        let keep = try XCTUnwrap(EditPlan(json: Self.json)?.keep)
        XCTAssertEqual(keep.boxes(at: 0), [EditPlan.Box(x: 0.1, y: 0.4, w: 0.2, h: 0.4)])
        let between = keep.boxes(at: 1.5)
        XCTAssertEqual(between.count, 1); XCTAssertEqual(between[0].x, 0.2, accuracy: 1e-9); XCTAssertEqual(between[0].w, 0.2, accuracy: 1e-9)
        XCTAssertEqual(keep.boxes(at: 2.4).count, 1)       // One box, then two: the nearer sample, not a blend.
        XCTAssertEqual(keep.boxes(at: 2.6).count, 2)
        XCTAssertEqual(keep.boxes(at: 99).count, 2)
    }

    func testPlateTimesAndKeepSamplesOutsideTheAttemptAreKept() throws {
        // The attempt is 3–6 s of a ten-second clip. The wall is also looked at before and after it, and the
        // box that must stay live is known for the whole stretch the person was followed, more sparsely outside.
        let times = (0..<20).map { Double($0) * 0.5 + 0.25 }
        let samples = [0.5, 1.5, 2.5] + (0...30).map { 3 + Double($0) * 0.1 } + [7, 8, 9]
        let boxes = samples.map { #"{"t":\#($0),"b":[[\#($0 / 20),0.4,0.2,0.4]]}"# }.joined(separator: ",")
        let json = #"{"v":1,"duration":10,"trim":[3,6],"segments":[{"t0":3,"t1":6,"speed":1}],"plate":{"times":\#(times)},"keep":{"feather":0.03,"boxes":[\#(boxes)]}}"#
        let plan = try XCTUnwrap(EditPlan(json: json))
        XCTAssertEqual(plan.plateTimes, times)                                          // All twenty, six of them before the attempt and seven after.
        XCTAssertEqual(plan.plateTimes.filter { !plan.trim.contains($0) }.count, 14)
        let keep = try XCTUnwrap(plan.keep)
        XCTAssertEqual(keep.samples.count, samples.count)
        XCTAssertEqual(keep.boxes(at: 0.25), [EditPlan.Box(x: 0.025, y: 0.4, w: 0.2, h: 0.4)])   // Before the first sample: the first.
        XCTAssertEqual(keep.boxes(at: 9.75), [EditPlan.Box(x: 0.45, y: 0.4, w: 0.2, h: 0.4)])    // After the last: the last.
        XCTAssertEqual(try XCTUnwrap(keep.boxes(at: 2.0).first).x, 0.1, accuracy: 1e-9)          // Between two sparse ones: in between.
        XCTAssertEqual(try XCTUnwrap(keep.boxes(at: 7.5).first).x, 0.375, accuracy: 1e-9)
        // A plan as large as a long attempt gives (tens of kilobytes) is read like any other.
        let long = (0..<1200).map { #"{"t":\#(Double($0) / 10),"b":[[0.3123,0.4123,0.2123,0.4123]]}"# }.joined(separator: ",")
        let large = #"{"v":1,"duration":120,"trim":[3,110],"segments":[{"t0":3,"t1":110,"speed":1}],"plate":{"times":\#((0..<25).map { Double($0) * 4.5 })},"keep":{"feather":0.03,"boxes":[\#(long)]}}"#
        XCTAssertGreaterThan(large.utf8.count, 40_000)
        let big = try XCTUnwrap(EditPlan(json: large))
        XCTAssertEqual(big.keep?.samples.count, 1200); XCTAssertEqual(big.plateTimes.count, 25)
    }

    func testPeopleAreFollowedFromFrameToFrameAndFailAsAWhole() {
        func box(_ x: Double, _ y: Double = 0.6) -> EditPlan.Box { EditPlan.Box(x: x, y: y, w: 0.1, h: 0.3) }
        var tracks = EditTracks(gap: 0.5)
        // Two people: one walks to the right, one stands. Their order in a frame does not matter.
        XCTAssertEqual(tracks.add(time: 0.0, boxes: [box(0.10), box(0.70)]), [0, 1])
        XCTAssertEqual(tracks.add(time: 0.1, boxes: [box(0.70), box(0.13)]), [1, 0])
        XCTAssertEqual(tracks.add(time: 0.2, boxes: [box(0.16)]), [0])                 // The one standing is not found this time.
        XCTAssertEqual(tracks.add(time: 0.3, boxes: [box(0.19), box(0.70), box(0.40, 0.1)]), [0, 1, 2])   // Found again: the same track. Someone new: a new one.
        XCTAssertEqual(tracks.tracks.map(\.samples.count), [4, 3, 1])
        // Unseen for longer than the gap: not the same track any more, even in the same place.
        XCTAssertEqual(tracks.add(time: 1.0, boxes: [box(0.70)]), [3])
        // No longer overlapping but close (a fast walker): still the same person.
        XCTAssertEqual(tracks.add(time: 1.1, boxes: [box(0.82)]), [3])
        XCTAssertEqual(tracks.add(time: 1.2, boxes: [box(0.20)]), [4])                 // Far away: someone else.

        // One moment that is not clean fails the whole track, before and after.
        tracks.setShape(7, track: 0)
        tracks.fail(1)
        XCTAssertEqual(tracks.tracks.map(\.removable), [true, false, true, true, true])
        XCTAssertEqual(tracks.shapes(at: 0.1, step: 0.1).map(\.track), [0])            // The standing one is not among the removable at any time.
        XCTAssertEqual(tracks.shapes(at: 0.3, step: 0.1).map(\.track), [0, 2])
        XCTAssertEqual(tracks.shapes(at: 0.3, step: 0.1).first?.shape, 7)
        XCTAssertEqual(tracks.shapes(at: 0.6, step: 0.1).map(\.track), [])             // Nobody is seen then.
        tracks.fail(99)                                                                 // Not a track: nothing happens.
    }

    func testAMomentWithoutADetectionInsideATrackIsBridged() {
        var tracks = EditTracks()
        let step = 1.0 / 30
        // Seen at frames 0, 1, 2, then not at 3, then at 4 and 5.
        for frame in [0, 1, 2, 4, 5] { _ = tracks.add(time: Double(frame) * step, boxes: [EditPlan.Box(x: 0.1 + 0.02 * Double(frame), y: 0.5, w: 0.1, h: 0.2)]) }
        XCTAssertEqual(tracks.tracks.count, 1)
        let found = tracks.shapes(at: 2 * step, step: step)
        XCTAssertEqual(found.map(\.bridged), [false]); XCTAssertEqual(try XCTUnwrap(found.first).box.x, 0.14, accuracy: 1e-9)
        let bridged = tracks.shapes(at: 3 * step, step: step)
        XCTAssertEqual(bridged.map(\.bridged), [true]); XCTAssertEqual(try XCTUnwrap(bridged.first).box.x, 0.16, accuracy: 1e-9)   // Halfway between the neighbours.
        XCTAssertNil(bridged.first?.shape)
        // A frame of a faster clip that falls between two looked-at moments takes the nearer one's outline.
        XCTAssertEqual(tracks.shapes(at: 1.4 * step, step: step).first?.box.x ?? 0, 0.12, accuracy: 1e-9)
        // The stand-in moments are listed so that they can be checked like the others.
        let bridges = tracks.bridges(step: step)
        XCTAssertEqual(bridges.count, 1); XCTAssertEqual(bridges.first?.time ?? 0, 3 * step, accuracy: 1e-9); XCTAssertEqual(bridges.first?.box.x ?? 0, 0.16, accuracy: 1e-9)
        // Before the first and after the last sighting there is nobody to replace; a failed track has no bridges.
        XCTAssertTrue(tracks.shapes(at: -2 * step, step: step).isEmpty); XCTAssertTrue(tracks.shapes(at: 7 * step, step: step).isEmpty)
        tracks.fail(0)
        XCTAssertTrue(tracks.bridges(step: step).isEmpty); XCTAssertTrue(tracks.shapes(at: 3 * step, step: step).isEmpty)
    }

    func testAPersonLostSightOfInTheMiddleOfThePictureIsNotTakenOutForTheRest() {
        func box(_ x: Double) -> EditPlan.Box { EditPlan.Box(x: x, y: 0.6, w: 0.1, h: 0.3) }
        let step = 0.1
        var tracks = EditTracks(gap: 0.25)
        // 0: walks in from the left edge and out of the looked-at stretch at its end. 1: there from the start, lost in the middle.
        // 2: appears in the middle (the same person found again, too late to be the same track), walks off at the right edge.
        // 3: appears and vanishes in the middle (a false sighting).
        for frame in 0...20 {
            let time = 1 + Double(frame) * step
            var boxes = [box(0.0 + 0.02 * Double(frame))]
            if frame <= 6 { boxes.append(box(0.3 + 0.01 * Double(frame)).grown(by: -0.01)) }
            if frame >= 12 { boxes.append(box(min(0.9, 0.6 + 0.03 * Double(frame - 12)))) }
            if frame == 9 { boxes.append(EditPlan.Box(x: 0.5, y: 0.1, w: 0.05, h: 0.05)) }
            _ = tracks.add(time: time, boxes: boxes)
        }
        XCTAssertEqual(tracks.tracks.count, 4)
        tracks.failUnseenComingOrGoing(ranges: [(1, 3)], step: step)
        XCTAssertEqual(tracks.tracks.map(\.removable), [true, false, false, false])
        // Coming in at an edge and leaving at an edge, well inside the stretch, is seen coming and going.
        var crossing = EditTracks()
        for frame in 0...10 { _ = crossing.add(time: 5 + Double(frame) * step, boxes: [box(Double(frame) * 0.09)]) }
        crossing.failUnseenComingOrGoing(ranges: [(0, 20)], step: step)
        XCTAssertEqual(crossing.tracks.map(\.removable), [true])
    }

    func testWhoeverWouldCoverAKeptPersonIsKeptTooAndSoOnDownTheLine() {
        func box(_ x: Double) -> EditPlan.Box { EditPlan.Box(x: x, y: 0.5, w: 0.1, h: 0.3) }
        let step = 0.1
        var tracks = EditTracks()
        // 0 stands at 0.50 and is kept. 1 walks from the left and passes over 0 around frames 8…12.
        // 2 stands at 0.05, where 1 was at the start: it never comes near 0, only near 1. 3 is far from everyone.
        for frame in 0...20 {
            _ = tracks.add(time: Double(frame) * step, boxes: [box(0.50), box(0.10 + 0.04 * Double(frame)), EditPlan.Box(x: 0.05, y: 0.75, w: 0.1, h: 0.2), EditPlan.Box(x: 0.85, y: 0.05, w: 0.1, h: 0.1)])
        }
        XCTAssertEqual(tracks.tracks.count, 4)
        tracks.fail(0)
        tracks.failThoseOverKeptPeople(step: step)
        // 1 would cover 0, so it stays; 2 would then cover 1, so it stays as well; 3 touches nobody and goes.
        XCTAssertEqual(tracks.tracks.map(\.removable), [false, false, false, true])

        // With nobody kept, nobody is held back, however much they overlap each other.
        var free = EditTracks()
        for frame in 0...5 { _ = free.add(time: Double(frame) * step, boxes: [box(0.5), EditPlan.Box(x: 0.52, y: 0.1, w: 0.1, h: 0.5)]) }
        free.failThoseOverKeptPeople(step: step)
        XCTAssertEqual(free.tracks.map(\.removable), [true, true])

        // A moment at which the removable person was not found counts too: the box that stands in for them
        // passes over the kept person although no sighting does.
        var bridged = EditTracks()
        let standing = EditPlan.Box(x: 0.51, y: 0.5, w: 0.08, h: 0.3)
        for frame in 0...10 {
            var boxes = [standing]
            if frame <= 3 || frame >= 7 { boxes.append(EditPlan.Box(x: 0.32 + 0.04 * Double(frame), y: 0.5, w: 0.05, h: 0.3)) }   // Unseen for three frames, right over the one standing.
            _ = bridged.add(time: Double(frame) * step, boxes: boxes)
        }
        XCTAssertEqual(bridged.tracks.count, 2)
        XCTAssertTrue(bridged.tracks[1].samples.allSatisfy { !$0.box.intersects(standing) })
        bridged.fail(0)
        bridged.failThoseOverKeptPeople(step: step)
        XCTAssertEqual(bridged.tracks.map(\.removable), [false, false])
        // A kept person is only in the way while they are there.
        XCTAssertNil(tracks.shape(of: 0, at: 9, step: step)); XCTAssertNotNil(tracks.shape(of: 0, at: 1.04, step: step))
    }

    func testGradeScaleFollowsTheDiscipline() {
        XCTAssertEqual(GradeScale.grades(for: nil), GradeScale.boulder)
        XCTAssertEqual(GradeScale.grades(for: .boulder).first, "V0"); XCTAssertEqual(GradeScale.grades(for: .boulder).last, "V16")
        XCTAssertEqual(GradeScale.grades(for: .toprope), GradeScale.grades(for: .lead))
        XCTAssertEqual(Array(GradeScale.rope.prefix(7)), ["5.5", "5.6", "5.7", "5.8", "5.9", "5.10a", "5.10b"])
        XCTAssertEqual(GradeScale.rope.last, "5.14d"); XCTAssertEqual(GradeScale.rope.count, 25)
        // A grade of the other scale is cleared; ungraded and anything an older build wrote are kept.
        XCTAssertEqual(GradeScale.kept("V4", for: .lead), ungradedValue)
        XCTAssertEqual(GradeScale.kept("5.10a", for: .boulder), ungradedValue)
        XCTAssertEqual(GradeScale.kept("5.10a", for: nil), ungradedValue)
        XCTAssertEqual(GradeScale.kept("V4", for: .boulder), "V4")
        XCTAssertEqual(GradeScale.kept("5.9", for: .toprope), "5.9")
        XCTAssertEqual(GradeScale.kept(ungradedValue, for: .lead), ungradedValue)
        XCTAssertEqual(GradeScale.kept("6a+", for: .lead), "6a+")
    }

    func testDisciplineGuessOnlyAcceptsTheThreeKinds() {
        XCTAssertEqual(DisciplineGuess(body: ["kind": "lead", "sure": true]), DisciplineGuess(kind: .lead, sure: true))
        XCTAssertEqual(DisciplineGuess(body: ["kind": "toprope"]), DisciplineGuess(kind: .toprope, sure: false))
        XCTAssertNil(DisciplineGuess(body: ["kind": "speed", "sure": true]))
        XCTAssertNil(DisciplineGuess(body: "boulder")); XCTAssertNil(DisciplineGuess(body: nil)); XCTAssertNil(DisciplineGuess(body: ["sure": true]))
    }

    func testRecordsWithoutADisciplineStillReadAndUnknownOnesAreNotAChoice() throws {
        let old = #"{"id":"7F9619FF-8B86-D011-B42D-00C04FC964FF","date":0,"title":"","grade":"V3","sent":false,"note":"","videoName":"a.mov","thumbnailName":"a.jpg","analysis":{"frames":[],"sampledFrames":0,"duration":1,"aspectRatio":1}}"#
        var climb = try JSONDecoder().decode(Climb.self, from: Data(old.utf8))
        XCTAssertNil(climb.style); XCTAssertNil(climb.chosenStyle)
        climb.discipline = "lead"
        XCTAssertEqual(climb.chosenStyle, .lead)
        climb.disciplineAuto = true
        XCTAssertEqual(climb.style, .lead); XCTAssertNil(climb.chosenStyle)       // A guess is not handed back as a choice.
        climb.discipline = "something newer"
        let again = try JSONDecoder().decode(Climb.self, from: JSONEncoder().encode(climb))
        XCTAssertNil(again.style); XCTAssertEqual(again.discipline, "something newer")
    }
}
