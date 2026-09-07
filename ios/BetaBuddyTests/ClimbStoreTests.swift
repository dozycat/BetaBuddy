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
        XCTAssertThrowsError(try store.complete(Drill.all[0]))
        XCTAssertEqual(try Data(contentsOf:file),original)
    }
    func testDrillCompletionIsIdempotent() throws {
        let store = ClimbStore(root:root)
        try store.complete(Drill.all[0]); try store.complete(Drill.all[0])
        XCTAssertEqual(ClimbStore(root:root).archive.training.count,1)
    }
    func testBundledFullModelIsPresent() throws {
        let url = try XCTUnwrap(Bundle.main.url(forResource:"pose_landmarker_full",withExtension:"task"))
        XCTAssertEqual(try Data(contentsOf:url).count,9398198)
    }
}
