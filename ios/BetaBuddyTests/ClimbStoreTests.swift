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
        let analysis = PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1)
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
    func testFoldersAndMetadataPersistAndFolderDeletionKeepsMedia() throws {
        let store = ClimbStore(root: root)
        try store.saveFolder(name: " Projects ")
        let folder = try XCTUnwrap(store.folders.first)
        XCTAssertEqual(folder.name, "Projects")
        XCTAssertThrowsError(try store.saveFolder(name: "projects"))
        XCTAssertThrowsError(try store.saveFolder(name: "  "))
        let staged = root.appendingPathComponent("fixture.mov")
        let video = Data([0, 1, 2]); try video.write(to: staged)
        let analysis = PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1)
        let climb = try store.save(title: "test", grade: "V2", sent: false, note: "keep", stagedVideo: staged, thumbnail: Data(), analysis: analysis, gymId: "fixture", gymName: "Fixture Gym", climbedAt: Date(timeIntervalSince1970: 1000), folderId: folder.id)
        try store.saveFolder(name: "Next try", id: folder.id)
        try store.organize(climb, title: "edited", grade: "V3", gymId: nil, gymName: "Handwritten", climbedAt: Date(timeIntervalSince1970: 2000), folderId: folder.id)
        let reopened = ClimbStore(root: root)
        XCTAssertEqual(reopened.folders.first?.name, "Next try")
        let edited = try XCTUnwrap(reopened.climbs.first)
        XCTAssertEqual(edited.gymName, "Handwritten")
        XCTAssertNil(edited.gymId)
        XCTAssertEqual(edited.climbingDate, Date(timeIntervalSince1970: 2000))
        XCTAssertEqual(edited.date, climb.date)
        XCTAssertEqual(edited.note, "keep")
        XCTAssertEqual(edited.folderId, folder.id)
        try reopened.deleteFolder(folder)
        let afterDelete = ClimbStore(root: root)
        XCTAssertTrue(afterDelete.folders.isEmpty)
        XCTAssertNil(afterDelete.climbs.first?.folderId)
        XCTAssertEqual(afterDelete.climbs.count, 1)
        XCTAssertEqual(try Data(contentsOf: store.file(climb.videoName)), video)
    }
    func testFailedFolderWriteLeavesPublishedArchiveUnchanged() throws {
        let store = ClimbStore(root: root)
        try store.saveFolder(name: "Keep")
        try FileManager.default.removeItem(at: root)
        XCTAssertThrowsError(try store.saveFolder(name: "Cannot save"))
        XCTAssertEqual(store.folders.map(\.name), ["Keep"])
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
        let analysis = PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1)
        XCTAssertThrowsError(try store.save(title:"",grade:ungradedValue,sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:analysis))
        XCTAssertEqual(try Data(contentsOf:file),original)
        XCTAssertTrue(FileManager.default.fileExists(atPath:staged.path))
    }
    // Archives written by earlier builds carry a `training` array and the retired review cues; both are ignored, not an error.
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
        XCTAssertEqual(climb.analysis.sampledFrames,4)
        XCTAssertFalse(climb.analysis.hasWorldLandmarks)   // needs pose recognition again before the analysis page
        try store.update(climb,note:"改过",sent:false)
        let rewritten = try String(contentsOf:root.appendingPathComponent("archive.json"),encoding:.utf8)
        XCTAssertFalse(rewritten.contains("training"))
        XCTAssertFalse(rewritten.contains("moments"))
        XCTAssertEqual(ClimbStore(root:root).climbs.first?.note,"改过")
    }
    // The analysis page's result is an opaque string kept next to the clip; synthetic here.
    func testAnalysisResultIsSavedReplacedReloadedAndDeletedWithTheRecord() throws {
        let suite = "store-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName:suite))
        defer { defaults.removePersistentDomain(forName:suite) }
        let store = ClimbStore(root:root,defaults:defaults)
        let staged = root.appendingPathComponent("fixture.mov")
        try Data([0,1,2]).write(to:staged)
        let climb = try store.save(title:"t",grade:"V1",sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:PoseAnalysis(frames:[],sampledFrames:0,duration:1,aspectRatio:1))
        XCTAssertNil(store.analysisResult(for:climb))
        XCTAssertFalse(store.analysisResults.contains(climb.id))
        let first = #"{"v":1,"note":"引号 \" 与换行 \n 原样保留"}"#
        try store.saveAnalysisResult(first,for:climb.id)
        XCTAssertEqual(store.analysisResult(for:climb),first)
        let file = root.appendingPathComponent("\(climb.id).analysis.json")
        XCTAssertEqual(try Data(contentsOf:file),Data(first.utf8))
        XCTAssertEqual(try root.resourceValues(forKeys:[.isExcludedFromBackupKey]).isExcludedFromBackup,true)
        try store.saveAnalysisResult(#"{"v":1,"second":true}"#,for:climb.id)
        let reopened = ClimbStore(root:root,defaults:defaults)
        XCTAssertTrue(reopened.analysisResults.contains(climb.id))
        XCTAssertEqual(reopened.analysisResult(for:climb),#"{"v":1,"second":true}"#)
        XCTAssertEqual(reopened.climbs.count,1)
        // “重新分析” drops the saved result and nothing else.
        reopened.discardAnalysisResult(climb)
        XCTAssertNil(reopened.analysisResult(for:climb))
        XCTAssertFalse(FileManager.default.fileExists(atPath:file.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath:reopened.file(climb.videoName).path))
        try reopened.saveAnalysisResult(first,for:climb.id)
        try reopened.delete(climb)
        XCTAssertFalse(FileManager.default.fileExists(atPath:file.path))
        XCTAssertTrue(reopened.analysisResults.isEmpty)
        // A result that arrives after the record is gone is not written.
        try reopened.saveAnalysisResult(first,for:climb.id)
        XCTAssertFalse(FileManager.default.fileExists(atPath:file.path))
        XCTAssertTrue(ClimbStore(root:root,defaults:defaults).analysisResults.isEmpty)
    }
    func testAnalysisClimberIsRememberedOnThisDevice() throws {
        let suite = "store-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName:suite))
        defer { defaults.removePersistentDomain(forName:suite) }
        let store = ClimbStore(root:root,defaults:defaults)
        XCTAssertNil(store.analysisClimber)
        store.analysisClimber = ["height":172,"weight":63.5]
        XCTAssertEqual(ClimbStore(root:root,defaults:defaults).analysisClimber,["height":172,"weight":63.5])
    }
    func testBundledFullModelIsPresent() throws {
        let url = try XCTUnwrap(Bundle.main.url(forResource:"pose_landmarker_full",withExtension:"task"))
        XCTAssertEqual(try Data(contentsOf:url).count,9398198)
    }
}
