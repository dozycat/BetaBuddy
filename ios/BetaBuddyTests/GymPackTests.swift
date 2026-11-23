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

    private func multiVenuePack() throws -> GymPack {
        let json = manifest.replacingOccurrences(of: #""schema":1"#, with: #""schema":1,"venues":[{"id":"banana-kerry","files":["index.html"]},{"id":"dome-xuhui","files":["data/scene.bin"]}]"#)
        return try GymPack(directory: writePack(manifest: json))
    }

    func testAdditionalVenueGateCoversLinksAndResources() throws {
        let pack = try multiVenuePack()
        for suffix in ["?venue=dome-xuhui", "?route=dome-xuhui%2Froute", "?wall=dome-xuhui%2Froof", "?venue=banana-kerry&route=dome-xuhui/r", "?venue=future-gym", "?route=/malformed"] {
            XCTAssertTrue(pack.manifest.requiresMembership(for: URL(string: "gympack://local/index.html" + suffix)!), suffix)
        }
        for suffix in ["", "?venue=banana-kerry", "?route=baishan-green", "?wall=banana-kerry/baishan"] {
            XCTAssertFalse(pack.manifest.requiresMembership(for: URL(string: "gympack://local/index.html" + suffix)!), suffix)
        }
        var member = false
        let handler = GymSchemeHandler(pack: pack, isMember: { member })
        let view = WKWebView()
        for path in ["data/scene.bin", "data/%73cene.bin", "data/../data/scene.bin"] {
            let task = SchemeTask("gympack://local/" + path)
            handler.webView(view, start: task)
            XCTAssertEqual(task.response?.statusCode, 403)
            XCTAssertNotEqual(task.body, Data("fixture".utf8))
        }
        member = true
        let unlocked = SchemeTask("gympack://local/data/scene.bin")
        handler.webView(view, start: unlocked)
        XCTAssertEqual(unlocked.response?.statusCode, 200)
        member = false
        let revoked = SchemeTask("gympack://local/data/scene.bin")
        handler.webView(view, start: revoked)
        XCTAssertEqual(revoked.response?.statusCode, 403)
    }

    func testCatGateCoversManifestAssetsURLsAndRevocation() throws {
        let assets = ["index.html", "cat.js", "catplans.0.js", "data/custom-cat.js", "data/custom-plans.js"]
        let json: [String: Any] = ["schema": 1, "packVersion": "1.8.0", "generatedAt": "2026-11-21", "gymName": "Fixture", "routeCount": 1,
            "catCode": "data/custom-cat.js", "venues": [["id": "banana-kerry", "files": ["index.html"], "catFiles": ["data/custom-plans.js"]]],
            "files": assets.map { ["path": $0, "sha256": "aa"] }]
        let pack = try GymPack(directory: writePack(manifest: String(decoding: JSONSerialization.data(withJSONObject: json), as: UTF8.self), files: assets))
        var member = false
        let handler = GymSchemeHandler(pack: pack, isMember: { member })
        let web = WKWebView()
        for allowed in [false, true, false] {
            member = allowed
            for path in assets.dropFirst() + ["index.html?species=cat", "data/%63ustom-cat.js"] {
                let task = SchemeTask("gympack://local/" + path)
                handler.webView(web, start: task)
                XCTAssertEqual(task.response?.statusCode, allowed ? 200 : 403, path)
            }
            let human = SchemeTask("gympack://local/index.html?species=human")
            handler.webView(web, start: human)
            XCTAssertEqual(human.response?.statusCode, 200)
        }
        let session = GymSession(status: .ready(pack))
        session.requestCat(preset: 1, strict: true)
        XCTAssertTrue(session.showingMembership)
        XCTAssertEqual(session.pendingCat?.preset, 1)
        session.cancelMembershipRequest()
        XCTAssertNil(session.pendingCat)
    }

    func testCancelledPaywallDoesNotOpenPremiumVenue() throws {
        let session = GymSession(status: .ready(try multiVenuePack()))
        let premium = URL(string: "gympack://local/index.html?venue=dome-xuhui")!
        XCTAssertFalse(session.authorize(premium))
        XCTAssertTrue(session.showingMembership)
        XCTAssertEqual(session.pendingMemberURL, premium)
        session.cancelMembershipRequest()
        XCTAssertNil(session.pendingMemberURL)
        XCTAssertFalse(session.authorize(premium))
        session.updateMembership(true)
        XCTAssertNil(session.pendingMemberURL)
        XCTAssertFalse(session.showingMembership)
        XCTAssertTrue(session.authorize(premium))
        session.updateMembership(false)
        XCTAssertFalse(session.authorize(premium))
    }

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
        XCTAssertEqual(pack.entryURL(hintSeen:false,language:AppLanguage.resolve(selection:"ko",system:["en-US"])).absoluteString,
                       "gympack://local/index.html?lang=ko")
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
        XCTAssertEqual(GymMessage(body:["type":"openSettings"]),.openSettings)
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
    // The analysis page is offered only by a pack that lists it.
    func testAnalysisPageIsOfferedOnlyWhenThePackListsIt() throws {
        let without = try GymPack(directory:try writePack(manifest:manifest))
        XCTAssertFalse(without.supportsAnalysis)
        XCTAssertNil(without.analysisURL(language:"en"))
        let listed = manifest.replacingOccurrences(of:#""files":["#,with:#""files":[{"path":"analysis.html","sha256":"cc","bytes":7},"#)
        let with = try GymPack(directory:try writePack(manifest:listed,files:["index.html","data/scene.bin","analysis.html"]))
        XCTAssertTrue(with.supportsAnalysis)
        let url = try XCTUnwrap(with.analysisURL(language:"zh-Hant"))
        XCTAssertEqual(url.absoluteString,"gympack://local/analysis.html?lang=zh-Hant")
        XCTAssertTrue(with.allows(url))
        XCTAssertEqual(with.file(for:url)?.lastPathComponent,"analysis.html")
        // Listed but not on disk: the pack as a whole is incomplete, as for any other file.
        try FileManager.default.removeItem(at:root.appendingPathComponent("GymPack/analysis.html"))
        XCTAssertEqual(GymPack.status(in:root),.invalid(.missingFile("analysis.html")))
    }
    func testAnalysisPageMessages() {
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisReady","packVersion":"0.0.1"]),.ready)
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisResult","id":"abc","json":"{\"v\":1}","climber":["height":171,"weight":62.5,"flag":true,"name":"x","bad":Double.nan]]),
                       .result(id:"abc",json:"{\"v\":1}",climber:["height":171,"weight":62.5]))
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisResult","json":"{}"]),.result(id:"",json:"{}",climber:[:]))
        // Whether the capture plan was used, when the page says so; anything else about it is ignored.
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisResult","json":"{}","capture":["used":true,"start":"user","finish":"video","extra":1]]),
                       .result(id:"",json:"{}",climber:[:],capture:CapturePlanUse(used:true,start:"user",finish:"video")))
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisResult","json":"{}","capture":["used":false,"reason":"mismatch"]]),
                       .result(id:"",json:"{}",climber:[:],capture:CapturePlanUse(used:false,reason:"mismatch")))
        for junk in [["reason":"mismatch"], ["used":"yes"], "used", NSNull()] as [Any] {
            XCTAssertEqual(AnalysisMessage(body:["type":"analysisResult","json":"{}","capture":junk]),.result(id:"",json:"{}",climber:[:]))
        }
        XCTAssertNil(AnalysisMessage(body:["type":"analysisResult","id":"abc"]))
        XCTAssertNil(AnalysisMessage(body:["type":"analysisResult","json":""]))
        XCTAssertNil(AnalysisMessage(body:["type":"analysisResult","json":String(repeating:"x",count:AnalysisMessage.maximumResultLength + 1)]))
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisSeek","t":12.5]),.seek(12.5))
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisSeek","t":-3]),.seek(0))
        XCTAssertNil(AnalysisMessage(body:["type":"analysisSeek"]))
        XCTAssertNil(AnalysisMessage(body:["type":"analysisSeek","t":"soon"]))
        XCTAssertNil(AnalysisMessage(body:["type":"analysisSeek","t":Double.infinity]))
        XCTAssertEqual(AnalysisMessage(body:["type":"analysisError","code":"synthetic","message":"m"]),.failed(code:"synthetic"))
        XCTAssertEqual(AnalysisMessage(body:["type":"error","message":"boom"]),.pageError("boom"))
        // Progress is the page's own to show; unknown types and other shapes are ignored.
        XCTAssertNil(AnalysisMessage(body:["type":"analysisProgress","stage":"s","fraction":0.5]))
        XCTAssertNil(AnalysisMessage(body:["type":"somethingElse"]))
        XCTAssertNil(AnalysisMessage(body:"analysisReady"))
        // The gym page's messages are untouched, and neither parser takes the other's types.
        XCTAssertNil(GymMessage(body:["type":"analysisReady"]))
        XCTAssertNil(GymMessage(body:["type":"analysisSeek","t":1]))
        XCTAssertNil(AnalysisMessage(body:["type":"ready"]))
        XCTAssertNil(AnalysisMessage(body:["type":"routeSelected","id":"r-1"]))
    }
    func testAnalysisCommandsEncodeTheirArguments() throws {
        XCTAssertTrue(AnalysisCommand.run.script.contains("BetaBuddyAnalysis.run()"))
        let begin = AnalysisCommand.begin(id:"7F9619FF",width:1080,height:1920,climber:["height":171]).script
        XCTAssertTrue(begin.contains(#""id":"7F9619FF""#))
        XCTAssertTrue(begin.contains(#""w":1080"#))
        XCTAssertTrue(begin.contains(#""h":1920"#))
        XCTAssertTrue(begin.contains(#""climber":{"height":171}"#))
        // No remembered measurements: the key is left out so the page uses its own defaults.
        XCTAssertFalse(AnalysisCommand.begin(id:"a",width:1,height:1,climber:nil).script.contains("climber"))
        XCTAssertFalse(AnalysisCommand.begin(id:"a",width:1,height:1,climber:[:]).script.contains("climber"))
        XCTAssertTrue(AnalysisCommand.addFrame(index:3,time:1.23456,jpegBase64:"QUJD+/8=").script.contains(#"addFrame(3,1.2346,"QUJD+/8=")"#))
        // A saved result goes back as one string literal, whatever it contains.
        let hostile = "{\"a\":\"\\\"); alert(1); (\\\"\",\"b\":\"line\\nbreak \u{2028} </script>\"}"
        let show = AnalysisCommand.show(hostile).script
        let start = try XCTUnwrap(show.range(of:"BetaBuddyAnalysis.show(",options:.backwards))
        let end = try XCTUnwrap(show.range(of:"), true)"))
        let literal = String(show[start.upperBound..<end.lowerBound])
        XCTAssertEqual(try JSONDecoder().decode(String.self,from:Data(literal.utf8)),hostile)
        XCTAssertFalse(literal.dropFirst().dropLast().contains { $0 == "\n" })
        // Samples without world landmarks are not sent.
        var frame = PoseFrame(time:0.5,joints:Array(repeating:Joint(x:0.25,y:0.75,confidence:0.9),count:33))
        XCTAssertTrue(AnalysisCommand.addPoses([frame]).script.contains("addPoses([])"))
        frame.world = Array(repeating:[0.1,-0.2,0.3,0.8],count:33)
        let poses = AnalysisCommand.addPoses([frame]).script
        XCTAssertTrue(poses.contains(#"addPoses([{"#))
        XCTAssertTrue(poses.contains(#""t":0.5"#))
        XCTAssertTrue(poses.contains("[0.25,0.75,0.8]"))
        XCTAssertTrue(poses.contains("[0.1,-0.2,0.3,0.8]"))
    }
    func testOfflineRulesAreValidJSON() throws {
        let rules = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(GymSession.blockRules.utf8)) as? [[String:Any]])
        XCTAssertEqual(rules.count,3)
    }
}
