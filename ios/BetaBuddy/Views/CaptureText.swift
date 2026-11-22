import Foundation

/// Words for what the capture page reports as codes. The page's own text is never shown.
enum CaptureText {
    /// What is wrong, for a code this build knows; the generic line for any other.
    static func reason(_ code: String) -> String {
        let bundle = AppLanguage.bundle
        guard let known = CaptureGuidance.Code(rawValue: code) else { return String(localized: "换个机位试试", bundle: bundle) }
        switch known {
        case .noWall: return String(localized: "没认出攀岩墙，把镜头对着整面墙", bundle: bundle)
        case .tooDark: return String(localized: "画面太暗，换个亮一点的位置", bundle: bundle)
        case .backlit: return String(localized: "逆光或者过曝了", bundle: bundle)
        case .blurry: return String(localized: "画面是糊的，擦擦镜头再试", bundle: bundle)
        case .shaking: return String(localized: "手机还在晃", bundle: bundle)
        case .tilted: return String(localized: "手机歪了", bundle: bundle)
        case .tooClose: return String(localized: "离墙太近，拍不全", bundle: bundle)
        case .tooFar: return String(localized: "离墙太远，岩点太小认不清", bundle: bundle)
        case .cutTop: return String(localized: "上面出画了", bundle: bundle)
        case .cutBottom: return String(localized: "下面出画了", bundle: bundle)
        case .cutLeft: return String(localized: "左边出画了", bundle: bundle)
        case .cutRight: return String(localized: "右边出画了", bundle: bundle)
        case .oblique: return String(localized: "机位太偏，斜着拍透视太强", bundle: bundle)
        case .pitchHigh: return String(localized: "手机仰得太厉害", bundle: bundle)
        case .occluded: return String(localized: "墙前面有东西或有人挡着", bundle: bundle)
        case .noHolds: return String(localized: "没认出这个颜色的岩点", bundle: bundle)
        case .startUnclear: return String(localized: "起步点认不准，可以点岩点自己标", bundle: bundle)
        case .finishUnclear: return String(localized: "结束点认不准，可以点岩点自己标", bundle: bundle)
        }
    }

    /// What to do about it, or `nil` when the page gave no move or one this build has no words for.
    static func action(_ move: String?) -> String? {
        let bundle = AppLanguage.bundle
        guard let move, let known = CaptureGuidance.Move(rawValue: move) else { return nil }
        switch known {
        case .back: return String(localized: "往后退一点", bundle: bundle)
        case .closer: return String(localized: "靠近一点", bundle: bundle)
        case .left: return String(localized: "手机往左挪", bundle: bundle)
        case .right: return String(localized: "手机往右挪", bundle: bundle)
        case .up: return String(localized: "手机放高一点", bundle: bundle)
        case .down: return String(localized: "手机放低一点", bundle: bundle)
        case .tiltUp: return String(localized: "镜头朝上转一点", bundle: bundle)
        case .tiltDown: return String(localized: "镜头朝下转一点", bundle: bundle)
        case .level: return String(localized: "把手机摆正", bundle: bundle)
        case .holdStill: return String(localized: "放稳别动", bundle: bundle)
        }
    }

    /// The one line of guidance for an issue.
    static func line(_ issue: CaptureAssessment.Issue) -> String {
        let reason = reason(issue.code)
        guard let action = action(issue.move) else { return reason }
        return String(localized: "\(reason)，\(action)", bundle: AppLanguage.bundle)
    }

    /// Short names for what a tap marks; also the badges on marked holds.
    static func role(_ role: CaptureRole) -> String {
        let bundle = AppLanguage.bundle
        switch role {
        case .startHand: return String(localized: "起步手", bundle: bundle)
        case .startFoot: return String(localized: "起步脚", bundle: bundle)
        case .finish: return String(localized: "结束", bundle: bundle)
        case .hold: return String(localized: "算这条线", bundle: bundle)
        case .none: return String(localized: "不算", bundle: bundle)
        }
    }

    /// The same for a role the page only guessed: with a question mark.
    static func guess(_ role: CaptureRole) -> String {
        let name = Self.role(role)
        return String(localized: "\(name)？", bundle: AppLanguage.bundle)
    }

    /// Why a recording ended by itself; `nil` when the user stopped it.
    static func stop(_ reason: CaptureStop?) -> String? {
        let bundle = AppLanguage.bundle
        switch reason {
        case .limit: return String(localized: "录到 20 分钟，自动停了。", bundle: bundle)
        case .storage: return String(localized: "存储空间快满了，已保存录到的部分。", bundle: bundle)
        case .interrupted: return String(localized: "录像被打断了，已保存录到的部分。", bundle: bundle)
        case .overheated: return String(localized: "手机温度太高，已保存录到的部分。", bundle: bundle)
        case .user, nil: return nil
        }
    }

    static func notice(_ notice: CaptureSession.Notice) -> String {
        let bundle = AppLanguage.bundle
        switch notice {
        case .tooShort: return String(localized: "录得太短，没有保存。", bundle: bundle)
        case .lowStorage: return String(localized: "存储空间不足，腾出一些空间再录。", bundle: bundle)
        case .recorderFailed: return String(localized: "录像没能开始，请再试一次。", bundle: bundle)
        }
    }

    static func distance(_ meters: Double) -> String {
        Measurement(value: meters, unit: UnitLength.meters).formatted(.measurement(width: .abbreviated, usage: .road).locale(AppLanguage.locale))
    }

    /// How far a suggested gym is, as far as that can honestly be said; `nil` when it cannot.
    static func distance(_ gym: NearbyGym) -> String? {
        guard let meters = gym.displayMeters else { return nil }
        let text = distance(meters)
        return gym.shown == .approximate ? String(localized: "约 \(text)", bundle: AppLanguage.bundle) : text
    }
}
