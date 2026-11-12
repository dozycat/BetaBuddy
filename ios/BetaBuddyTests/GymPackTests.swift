import XCTest
@testable import BetaBuddy

// Fixtures here are synthetic. Real venue data never enters this repository.
@MainActor
final class GymPackTests: XCTestCase {
    var root: URL!
    override func setUp() { super.setUp(); root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }
    override func tearDown() { try? FileManager.default.removeItem(at:root); super.tearDown() }

    private func writePack(manifest: String, files: [String] = ["index.html", "data/scene.bin"]) throws -> URL {
        let pack = root.appendingPathComponent(GymPack.directoryName)
        for file in files {
            let url = pack.appendingPathComponent(file)
            try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
            try Data("fixture".utf8).write(to:url)
        }
        try FileManager.default.createDirectory(at:pack,withIntermediateDirectories:true)
        try Data(manifest.utf8).write(to:pack.appendingPathComponent("manifest.json"))
        return pack
    }
    private let manifest = """
    {"schema":1,"packVersion":"0.0.1","generatedAt":"2026-01-02T03:04:05Z","gymName":"测试岩馆","routeCount":7,
     "entry":"index.html","files":[{"path":"index.html","sha256":"aa","bytes":7},{"path":"data/scene.bin","sha256":"bb","bytes":7}]}
    """

    func testManifestDecodes() throws {
        let decoded = try JSONDecoder().decode(GymManifest.self, from:Data(manifest.utf8))
        XCTAssertEqual(decoded.packVersion,"0.0.1")
        XCTAssertEqual(decoded.generatedAt,"2026-01-02T03:04:05Z")
        XCTAssertEqual(decoded.gymName,"测试岩馆")
        XCTAssertEqual(decoded.routeCount,7)
        XCTAssertEqual(decoded.entry,"index.html")
        XCTAssertEqual(decoded.files.map(\.path),["data/scene.bin","index.html"])
    }
    func testManifestAcceptsSnakeCaseAndFileMap() throws {
        let json = #"{"pack_version":"2","generated_at":"2026-01-02","gym":{"name":"G"},"route_count":3,"files":{"index.html":"aa"}}"#
        let decoded = try JSONDecoder().decode(GymManifest.self, from:Data(json.utf8))
        XCTAssertEqual(decoded.schema,1)
        XCTAssertEqual(decoded.gymName,"G")
        XCTAssertEqual(decoded.routeCount,3)
        XCTAssertEqual(decoded.entry,"index.html")
        XCTAssertEqual(decoded.files,[.init(path:"index.html",sha256:"aa")])
    }
    func testManifestWithoutRequiredFieldIsRejected() {
        let json = #"{"packVersion":"1","generatedAt":"2026-01-02","routeCount":3,"files":[]}"#
        XCTAssertThrowsError(try JSONDecoder().decode(GymManifest.self, from:Data(json.utf8)))
    }

    func testNoPackDegradesToMissing() throws {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        XCTAssertEqual(GymPack.status(in:root),.missing)
        XCTAssertEqual(GymPack.status(in:nil),.missing)
        let session = GymSession(status:.missing)
        session.start()
        XCTAssertEqual(session.phase,.idle)
        XCTAssertNil(session.webView)
    }
    func testBundledStatusMatchesCurrent() {
        // Passes with or without an installed pack; a pack that is present must be usable.
        switch GymPack.status {
        case .missing: XCTAssertNil(GymPack.current)
        case .ready(let pack): XCTAssertEqual(GymPack.current,pack)
        case .invalid(let error): XCTFail("Bundled gym pack is broken: \(error)")
        }
    }
    func testValidPackLoads() throws {
        let directory = try writePack(manifest:manifest)
        guard case .ready(let pack) = GymPack.status(in:root) else { return XCTFail("pack should load") }
        XCTAssertEqual(pack.manifest.gymName,"测试岩馆")
        XCTAssertEqual(pack.entryURL.lastPathComponent,"index.html")
        XCTAssertTrue(pack.allows(directory.appendingPathComponent("data/scene.bin")))
        XCTAssertTrue(pack.allows(URL(string:pack.entryURL.absoluteString + "#route")))
        XCTAssertFalse(pack.allows(root.appendingPathComponent("outside.html")))
        XCTAssertFalse(pack.allows(directory.appendingPathComponent("../outside.html")))
        XCTAssertFalse(pack.allows(URL(string:"https://example.com/")))
        XCTAssertFalse(pack.allows(nil))
    }
    func testBrokenPacksAreInvalidNotMissing() throws {
        _ = try writePack(manifest:"not json")
        XCTAssertEqual(GymPack.status(in:root),.invalid(.unreadableManifest))
        _ = try writePack(manifest:manifest,files:["index.html"])
        try? FileManager.default.removeItem(at:root.appendingPathComponent("GymPack/data"))
        XCTAssertEqual(GymPack.status(in:root),.invalid(.missingFile("data/scene.bin")))
        _ = try writePack(manifest:manifest.replacingOccurrences(of:"\"schema\":1",with:"\"schema\":9"))
        XCTAssertEqual(GymPack.status(in:root),.invalid(.unsupportedSchema(9)))
        _ = try writePack(manifest:manifest.replacingOccurrences(of:"data/scene.bin",with:"../scene.bin"))
        XCTAssertEqual(GymPack.status(in:root),.invalid(.unsafePath("../scene.bin")))
    }

    func testPageMessages() {
        XCTAssertEqual(GymMessage(body:["type":"ready"]),.ready)
        XCTAssertEqual(GymMessage(body:["type":"routeSelected","id":"r-1","name":"紫色 3"]),.routeSelected(GymRoute(id:"r-1",name:"紫色 3")))
        XCTAssertEqual(GymMessage(body:["type":"routeSelected"]),.routeSelected(nil))
        XCTAssertEqual(GymMessage(body:["type":"error","message":"boom"]),.error("boom"))
        XCTAssertNil(GymMessage(body:["type":"somethingElse"]))
        XCTAssertNil(GymMessage(body:"ready"))
    }
    func testCommandsQuoteIdentifiers() {
        XCTAssertTrue(GymCommand.openRoute("r-1").script.contains(#"BetaBuddyGym.openRoute("r-1")"#))
        let hostile = GymCommand.openWall(#"a");alert(1);("#).script
        XCTAssertTrue(hostile.contains(#"openWall("a\");alert(1);(")"#))
    }
    func testOfflineRulesAreValidJSON() throws {
        let rules = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(GymSession.blockRules.utf8)) as? [[String:Any]])
        XCTAssertEqual(rules.count,3)
    }
}
