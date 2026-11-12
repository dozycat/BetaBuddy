import XCTest
@testable import BetaBuddy

@MainActor
final class ClimbStoreTests: XCTestCase {
    var root:URL!
    override func setUp() { super.setUp(); root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }
    override func tearDown() { try? FileManager.default.removeItem(at:root); super.tearDown() }
    func testSaveReloadEditAndDelete() throws {
        let store = ClimbStore(root:root)
        let staged = root.appendingPathComponent("fixture.mov")
        try Data([0,1,2]).write(to:staged)
        let analysis = PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1,processingSeconds:0,moments:[])
        let climb = try store.save(title:"测试",grade:"V3",sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:analysis)
        XCTAssertEqual(ClimbStore(root:root).climbs.count,1)
        try store.update(climb,note:"下一次看脚",sent:true)
        let reopened = ClimbStore(root:root)
        XCTAssertEqual(reopened.climbs.first?.note,"下一次看脚")
        XCTAssertEqual(reopened.climbs.first?.sent,true)
        var refreshedAnalysis = analysis
        refreshedAnalysis.sampledFrames = 42
        try reopened.replaceAnalysis(climb, analysis: refreshedAnalysis)
        let refreshed = try XCTUnwrap(ClimbStore(root: root).climbs.first)
        XCTAssertEqual(refreshed.analysis.sampledFrames, 42)
        XCTAssertEqual(refreshed.note, "下一次看脚")
        XCTAssertTrue(refreshed.sent)
        XCTAssertEqual(refreshed.id, climb.id)
        XCTAssertEqual(try Data(contentsOf: store.file(climb.videoName)), Data([0,1,2]))
        try reopened.delete(climb)
        XCTAssertTrue(ClimbStore(root:root).climbs.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath:store.file(climb.videoName).path))
    }
    func testCorruptArchiveIsNeverOverwritten() throws {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let file = root.appendingPathComponent("archive.json")
        let original = Data("broken".utf8)
        try original.write(to:file)
        let store = ClimbStore(root:root)
        XCTAssertNotNil(store.error)
        let staged = root.appendingPathComponent("fixture.mov")
        try Data([0,1,2]).write(to:staged)
        let analysis = PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1,processingSeconds:0,moments:[])
        XCTAssertThrowsError(try store.save(title:"",grade:ungradedValue,sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:analysis))
        XCTAssertEqual(try Data(contentsOf:file),original)
        XCTAssertTrue(FileManager.default.fileExists(atPath:staged.path))
    }
    // Archives written by builds that had practice logs carry a `training` array; it is ignored, not an error.
    func testArchiveWithRetiredFieldsStillLoads() throws {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let legacy = #"""
        {"version":1,"training":[{"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"drillID":"quiet-feet"}],
         "climbs":[{"id":"7F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"title":"旧记录","grade":"未标级","sent":true,"note":"n",
           "videoName":"v.mov","thumbnailName":"t.jpg",
           "analysis":{"frames":[],"sampledFrames":4,"duration":2,"aspectRatio":0.5,"processingSeconds":1,
             "moments":[{"id":"8F9619FF-8B86-D011-B42D-00C04FC964FF","time":1,"title":"高抬脚 · 候选片段","detail":"d"}]}}]}
        """#
        try Data(legacy.utf8).write(to:root.appendingPathComponent("archive.json"))
        let store = ClimbStore(root:root)
        XCTAssertNil(store.error)
        let climb = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(climb.title,"旧记录")
        XCTAssertNil(climb.analysis.moments[0].kind)
        XCTAssertNotEqual(climb.analysis.moments[0].displayDetail,"d")   // recognised by its stored title
        try store.update(climb,note:"改过",sent:false)
        let rewritten = try String(contentsOf:root.appendingPathComponent("archive.json"),encoding:.utf8)
        XCTAssertFalse(rewritten.contains("training"))
        XCTAssertEqual(ClimbStore(root:root).climbs.first?.note,"改过")
    }
    func testBundledFullModelIsPresent() throws {
        let url = try XCTUnwrap(Bundle.main.url(forResource:"pose_landmarker_full",withExtension:"task"))
        XCTAssertEqual(try Data(contentsOf:url).count,9398198)
    }
}
