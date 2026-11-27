import Vision

/// Generic image-region tracking. Coordinates entering and leaving this wrapper are top-left;
/// Vision's normalized rectangle is bottom-left. The private policy evaluates each observation.
final class SubjectBoxTracker {
    struct Observation { var bounds: SubjectBounds; var confidence: Double }
    private let request: VNTrackObjectRequest
    private let handler = VNSequenceRequestHandler()
    init(bounds: SubjectBounds) {
        let rectangle = CGRect(x: bounds.x, y: 1-bounds.y-bounds.height, width: bounds.width, height: bounds.height)
        request = VNTrackObjectRequest(detectedObjectObservation: VNDetectedObjectObservation(boundingBox: rectangle))
        request.revision = VNTrackObjectRequestRevision2
        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
    }
    func next(image: CGImage) throws -> Observation? {
        try handler.perform([request], on: image)
        guard let result = request.results?.first as? VNDetectedObjectObservation else { return nil }
        request.inputObservation = result
        let b = result.boundingBox
        let left = max(0,b.minX), right = min(1,b.maxX), top = max(0,1-b.maxY), bottom = min(1,1-b.minY)
        let bounds = SubjectBounds(x:left,y:top,width:right-left,height:bottom-top)
        guard bounds.valid else { return nil }
        return Observation(bounds: bounds, confidence: Double(result.confidence))
    }
}
