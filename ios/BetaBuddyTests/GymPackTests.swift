import XCTest
import WebKit
@testable import BetaBuddy

private final class SchemeTask: NSObject, WKURLSchemeTask {
    let request: URLRequest
    var response: HTTPURLResponse?
    var body = Data()
    var finished = false
    init(_ url: String) { request = URLRequest(url:URL(string:url)!) }
    func didReceive(_ response: URLResponse) { self.response = response as? HTTPURLResponse }
    func didReceive(_ data: Data) { body.append(data) }
    func didFinish() { finished = true }
    func didFailWithError(_ error: Error) { finished = true }
}

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
        XCTAssertEqual(pack.entryURL.absoluteString,"gympack://local/index.html")
        XCTAssertTrue(pack.allows(URL(string:"gympack://local/index.html#route")))
        XCTAssertFalse(pack.allows(URL(string:"gympack://elsewhere/index.html")))
        XCTAssertFalse(pack.allows(directory.appendingPathComponent("index.html")))
        XCTAssertFalse(pack.allows(URL(string:"https://example.com/")))
        XCTAssertFalse(pack.allows(nil))
    }
    func testPackURLsResolveOnlyInsideThePack() throws {
        let directory = try writePack(manifest:manifest)
        try Data("secret".utf8).write(to:root.appendingPathComponent("outside.html"))
        try FileManager.default.createSymbolicLink(at:directory.appendingPathComponent("link.html"),withDestinationURL:root.appendingPathComponent("outside.html"))
        let pack = try GymPack(directory:directory)
        func file(_ string: String) -> String? { pack.file(for:URL(string:string))?.lastPathComponent }
        XCTAssertEqual(file("gympack://local/index.html"),"index.html")
        XCTAssertEqual(file("gympack://local/data/scene.bin?v=1#x"),"scene.bin")
        XCTAssertEqual(file("gympack://local/data/../index.html"),"index.html")
        XCTAssertNil(file("gympack://local/../outside.html"))
        XCTAssertNil(file("gympack://local/data/../../outside.html"))
        XCTAssertNil(file("gympack://local/%2e%2e/outside.html"))
        XCTAssertNil(file("gympack://local/..%2Foutside.html"))
        XCTAssertNil(file("gympack://local/link.html"))
        XCTAssertNil(file("gympack://local/missing.js"))
        XCTAssertNil(file("gympack://local/data"))
        XCTAssertNil(file("gympack://local/"))
        XCTAssertNil(file("gympack://other/index.html"))
        XCTAssertNil(file("file://" + directory.appendingPathComponent("index.html").path))
        XCTAssertNil(file("https://local/index.html"))
    }
    func testMimeTypes() {
        let expected = ["html":"text/html; charset=utf-8","js":"text/javascript; charset=utf-8","mjs":"text/javascript; charset=utf-8",
                        "json":"application/json; charset=utf-8","css":"text/css; charset=utf-8","glb":"model/gltf-binary",
                        "bin":"application/octet-stream","wasm":"application/wasm","png":"image/png","jpg":"image/jpeg",
                        "svg":"image/svg+xml","txt":"text/plain; charset=utf-8","md":"text/markdown; charset=utf-8",
                        "GLB":"model/gltf-binary","unknown":"application/octet-stream"]
        for (pathExtension, type) in expected { XCTAssertEqual(GymPack.mimeType(forExtension:pathExtension),type,pathExtension) }
    }
    func testSchemeHandlerServesPackFilesAndRefusesTheRest() throws {
        let directory = try writePack(manifest:manifest)
        try Data("secret".utf8).write(to:root.appendingPathComponent("outside.html"))
        let handler = GymSchemeHandler(pack:try GymPack(directory:directory))
        let view = WKWebView()
        let served = SchemeTask("gympack://local/data/scene.bin")
        handler.webView(view,start:served)
        XCTAssertEqual(served.response?.statusCode,200)
        XCTAssertEqual(served.response?.value(forHTTPHeaderField:"Content-Type"),"application/octet-stream")
        XCTAssertEqual(served.response?.value(forHTTPHeaderField:"Cache-Control"),"no-store")
        XCTAssertEqual(served.response?.value(forHTTPHeaderField:"Content-Length"),"7")
        let policy = try XCTUnwrap(served.response?.value(forHTTPHeaderField:"Content-Security-Policy"))
        XCTAssertFalse(policy.contains("http"))
        XCTAssertFalse(policy.contains("*"))
        XCTAssertEqual(served.body,Data("fixture".utf8))
        XCTAssertTrue(served.finished)
        for refused in ["gympack://local/../outside.html","gympack://local/missing.js","gympack://other/index.html"] {
            let task = SchemeTask(refused)
            handler.webView(view,start:task)
            XCTAssertEqual(task.response?.statusCode,404,refused)
            XCTAssertNotEqual(task.body,Data("secret".utf8))
            XCTAssertTrue(task.finished)
        }
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

    func testEntryURLAssembly() throws {
        let pack = try GymPack(directory:try writePack(manifest:manifest))
        XCTAssertEqual(pack.entryURL(hintSeen:false).absoluteString,"gympack://local/index.html")
        XCTAssertEqual(pack.entryURL(hintSeen:true).absoluteString,"gympack://local/index.html?hint=0")
        XCTAssertEqual(pack.entryURL(hintSeen:true,extraQuery:"sheet=half&tab=steps&route=xxx&step=5").absoluteString,
                       "gympack://local/index.html?hint=0&sheet=half&tab=steps&route=xxx&step=5")
        XCTAssertEqual(pack.entryURL(hintSeen:false,extraQuery:"?wall=a").absoluteString,"gympack://local/index.html?wall=a")
        XCTAssertEqual(pack.entryURL(hintSeen:true,extraQuery:"").absoluteString,"gympack://local/index.html?hint=0")
        XCTAssertEqual(pack.entryURL(hintSeen:false,language:"ja").absoluteString,"gympack://local/index.html?lang=ja")
        XCTAssertEqual(pack.entryURL(hintSeen:true,language:"zh-Hant",extraQuery:"sheet=half").absoluteString,
                       "gympack://local/index.html?hint=0&lang=zh-Hant&sheet=half")
        for query in ["route=x#frag","a=1&b=%E7%BA%BF"] {
            let url = pack.entryURL(hintSeen:true,extraQuery:query)
            XCTAssertTrue(pack.allows(url))
            XCTAssertEqual(pack.file(for:url)?.lastPathComponent,"index.html")
        }
    }
    func testPageMessages() {
        XCTAssertEqual(GymMessage(body:["type":"ready"]),.ready)
        XCTAssertEqual(GymMessage(body:["type":"hintSeen"]),.hintSeen)
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
