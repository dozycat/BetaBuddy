import CoreLocation
import Foundation

/// What a directory entry's coordinates point at. Only the first three say where the gym is;
/// a district or city centre is the same point for every gym in it and can be tens of kilometres off.
enum CoordinatePrecision: String, CaseIterable, Sendable {
    case poi, address, station, district, city
    /// Good enough to rank gyms by distance.
    var ranks: Bool { self == .poi || self == .address || self == .station }
}

extension DirectoryGym {
    /// `nil` when the entry has none, or a value this build does not know: it then has no usable coordinate.
    var precision: CoordinatePrecision? { coordPrecision.flatMap(CoordinatePrecision.init(rawValue:)) }
    /// The entry's position when all three fields are there and make sense.
    var coordinate: (latitude: Double, longitude: Double)? {
        guard precision != nil, let lat, let lon, lat.isFinite, lon.isFinite, (-90...90).contains(lat), (-180...180).contains(lon),
              lat != 0 || lon != 0 else { return nil }
        return (lat, lon)
    }
}

/// A gym from the offline directory with how far away it is.
struct NearbyGym: Identifiable, Equatable {
    enum Shown: Equatable {
        /// The directory knows where this gym itself is.
        case exact
        /// Only the nearest station is known: shown rounded, as “about”.
        case approximate
        /// The phone's own position is too coarse for a number; the distance only orders the list.
        case hidden
    }
    var gym: DirectoryGym
    var meters: Double
    var shown: Shown
    var id: String { gym.id }
    /// The distance as it may be shown: as measured, or rounded to 100 m below a kilometre and to 0.5 km above.
    var displayMeters: Double? {
        switch shown {
        case .exact: meters
        case .approximate: meters < 1_000 ? max(100, (meters / 100).rounded() * 100) : (meters / 500).rounded() * 500
        case .hidden: nil
        }
    }
}

/// What the capture screen offers when the phone's position is known.
struct GymSuggestions: Equatable {
    /// Nearest first; only gyms whose own position is known.
    var ranked: [NearbyGym] = []
    /// Other gyms of the city the phone seems to be in, without distances: for these the directory
    /// only knows the district or the city, or nothing.
    var sameCity: [DirectoryGym] = []
    /// A gym close enough to be where the user is standing. Offered, never chosen for them.
    var here: NearbyGym?
}

enum GymDistance {
    /// Further than this is not “nearby”.
    static let reach: Double = 100_000
    /// A fix coarser than this cannot tell how far a gym is.
    static let usefulAccuracy: Double = 1_000
    /// Closer than this, with a fix at least this good, a gym is taken to be the one the user is at.
    static let arrival: Double = 300

    /// Great-circle distance in metres between two WGS-84 positions.
    static func meters(fromLatitude lat1: Double, longitude lon1: Double, toLatitude lat2: Double, longitude lon2: Double) -> Double {
        let radius = 6_371_008.8
        let p1 = lat1 * .pi / 180, p2 = lat2 * .pi / 180
        let dp = p2 - p1, dl = (lon2 - lon1) * .pi / 180
        let a = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
        return 2 * radius * asin(min(1, a.squareRoot()))
    }

    static func suggestions(_ gyms: [DirectoryGym], latitude: Double, longitude: Double, accuracy: Double, limit: Int = 6, cityLimit: Int = 12) -> GymSuggestions {
        guard latitude.isFinite, longitude.isFinite, (-90...90).contains(latitude), (-180...180).contains(longitude) else { return GymSuggestions() }
        let precise = accuracy >= 0 && accuracy <= usefulAccuracy
        // Every entry with a usable coordinate, however coarse, within reach.
        let located = gyms.compactMap { gym -> (gym: DirectoryGym, meters: Double, precision: CoordinatePrecision)? in
            guard let precision = gym.precision, let coordinate = gym.coordinate else { return nil }
            let meters = meters(fromLatitude: latitude, longitude: longitude, toLatitude: coordinate.latitude, longitude: coordinate.longitude)
            return meters <= reach ? (gym, meters, precision) : nil
        }.sorted { $0.meters != $1.meters ? $0.meters < $1.meters : $0.gym.name < $1.gym.name }
        let ranked = located.filter { $0.precision.ranks }.map { item in
            NearbyGym(gym: item.gym, meters: item.meters, shown: !precise ? .hidden : item.precision == .station ? .approximate : .exact)
        }
        var result = GymSuggestions(ranked: Array(ranked.prefix(limit)))
        // The city is read off the nearest entry of any kind: a district centre is no good for a
        // distance, but it does say which city this is.
        if let nearest = located.first {
            let shown = Set(result.ranked.map(\.gym.id))
            let coarse = Dictionary(located.map { ($0.gym.id, $0.meters) }, uniquingKeysWith: min)
            result.sameCity = Array(gyms.filter { $0.province == nearest.gym.province && $0.city == nearest.gym.city && !shown.contains($0.id) && $0.precision?.ranks != true }
                .sorted {
                    // Nearer districts first, entries without any coordinate last; never shown as a distance.
                    let a = coarse[$0.id] ?? .infinity, b = coarse[$1.id] ?? .infinity
                    return a != b ? a < b : $0.name < $1.name
                }.prefix(cityLimit))
        }
        if let first = ranked.first, first.shown == .exact, first.meters < arrival, accuracy <= arrival { result.here = first }
        return result
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

    /// Empty without a position.
    func suggestions() -> GymSuggestions {
        guard let fix else { return GymSuggestions() }
        return GymDistance.suggestions(GymDirectory.bundled?.gyms ?? [], latitude: fix.latitude, longitude: fix.longitude, accuracy: fix.accuracy)
    }
}
