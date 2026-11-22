import CoreLocation
import Foundation

/// A gym from the offline directory with how far away it is.
struct NearbyGym: Identifiable, Equatable {
    var gym: DirectoryGym
    var meters: Double
    /// The number is worth showing: the directory knows where this gym itself is, and the phone
    /// knows where it is to better than a kilometre. Otherwise the distance only orders the list.
    var showsDistance: Bool
    var id: String { gym.id }
}

enum GymDistance {
    /// Further than this is not “nearby”.
    static let reach: Double = 100_000
    /// A fix coarser than this cannot tell how far a gym is.
    static let usefulAccuracy: Double = 1_000

    /// Great-circle distance in metres between two WGS-84 positions.
    static func meters(fromLatitude lat1: Double, longitude lon1: Double, toLatitude lat2: Double, longitude lon2: Double) -> Double {
        let radius = 6_371_008.8
        let p1 = lat1 * .pi / 180, p2 = lat2 * .pi / 180
        let dp = p2 - p1, dl = (lon2 - lon1) * .pi / 180
        let a = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
        return 2 * radius * asin(min(1, a.squareRoot()))
    }

    /// Gyms that have coordinates, nearest first. Entries without usable coordinates are left out;
    /// they stay reachable through the full directory.
    static func nearby(_ gyms: [DirectoryGym], latitude: Double, longitude: Double, accuracy: Double, limit: Int = 6) -> [NearbyGym] {
        guard latitude.isFinite, longitude.isFinite, (-90...90).contains(latitude), (-180...180).contains(longitude) else { return [] }
        let located = gyms.compactMap { gym -> NearbyGym? in
            guard let lat = gym.lat, let lon = gym.lon, lat.isFinite, lon.isFinite, (-90...90).contains(lat), (-180...180).contains(lon),
                  lat != 0 || lon != 0 else { return nil }
            let meters = meters(fromLatitude: latitude, longitude: longitude, toLatitude: lat, longitude: lon)
            guard meters <= reach else { return nil }
            let exact = gym.coordPrecision == "poi" || gym.coordPrecision == "address"
            return NearbyGym(gym: gym, meters: meters, showsDistance: exact && accuracy >= 0 && accuracy <= usefulAccuracy)
        }
        return Array(located.sorted { $0.meters != $1.meters ? $0.meters < $1.meters : $0.gym.name < $1.gym.name }.prefix(limit))
    }
}

/// One position fix, taken only while the capture screen is choosing a gym. The position is used
/// to order the offline directory and is then forgotten: it is not stored and goes nowhere.
@MainActor
final class GymLocator: NSObject, ObservableObject, CLLocationManagerDelegate {
    enum Access: Equatable { case notAsked, allowed, refused }
    struct Fix: Equatable {
        var latitude: Double
        var longitude: Double
        var accuracy: Double
    }
    @Published private(set) var access = Access.notAsked
    @Published private(set) var fix: Fix?
    @Published private(set) var locating = false
    private var manager: CLLocationManager?

    override init() {
        super.init()
        #if DEBUG
        // `-debugLocation "<lat>,<lon>"` on the launch command line stands in for a fix, for tests. Nothing is asked.
        if let text = UserDefaults.standard.string(forKey: "debugLocation") {
            let parts = text.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            access = parts.count == 2 ? .allowed : .refused
            if parts.count == 2 { fix = Fix(latitude: parts[0], longitude: parts[1], accuracy: 30) }
            return
        }
        #endif
        let manager = CLLocationManager()
        manager.delegate = self
        // Enough to tell gyms apart; approximate location, if that is all the user allows, still orders the list.
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        self.manager = manager
        access = Self.access(manager.authorizationStatus)
    }

    private static func access(_ status: CLAuthorizationStatus) -> Access {
        switch status {
        case .notDetermined: .notAsked
        case .authorizedWhenInUse, .authorizedAlways: .allowed
        default: .refused
        }
    }

    /// Takes one fix when the user has already said yes. Never shows the system's question.
    func locateIfAllowed() {
        guard let manager, access == .allowed, fix == nil, !locating else { return }
        locating = true
        manager.requestLocation()
    }

    /// Shows the system's question if it has not been answered, then takes one fix.
    func ask() {
        guard let manager else { return }
        if access == .notAsked { manager.requestWhenInUseAuthorization() } else { locateIfAllowed() }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        Task { @MainActor in
            let before = access
            access = Self.access(status)
            if before == .notAsked, access == .allowed { locateIfAllowed() }
        }
    }
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        let fix = Fix(latitude: location.coordinate.latitude, longitude: location.coordinate.longitude, accuracy: location.horizontalAccuracy)
        Task { @MainActor in self.fix = fix; locating = false }
    }
    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // No position: the screen goes on with the last gym and the full directory.
        Task { @MainActor in locating = false }
    }

    func nearby(limit: Int = 6) -> [NearbyGym] {
        guard let fix else { return [] }
        return GymDistance.nearby(GymDirectory.bundled?.gyms ?? [], latitude: fix.latitude, longitude: fix.longitude, accuracy: fix.accuracy, limit: limit)
    }
}
