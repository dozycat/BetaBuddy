import SwiftUI

// Display text for values that are stored language-independently or in the source language.

/// Stored sentinel for a climb without a grade. Kept as is so existing archives stay readable.
let ungradedValue = "未标级"

func gradeText(_ grade: String) -> String {
    grade == ungradedValue ? String(localized: "未标级", bundle: AppLanguage.bundle) : grade
}

func percentText(_ fraction: Double) -> String {
    fraction.formatted(.percent.precision(.fractionLength(0)).locale(AppLanguage.locale))
}

extension Climb {
    /// A climb saved without a name is stored with an empty title and shown as “today's climb” in the
    /// current language. Titles that are exactly that default in any language (written by earlier
    /// builds at save time) are treated the same way, so they follow a language change too.
    var displayTitle: String {
        let name = String(localized: "今天的攀爬", bundle: AppLanguage.bundle)
        if title.isEmpty { return name }
        let defaults = AppLanguage.supported.map { String(localized: "今天的攀爬", bundle: AppLanguage.bundle(for: $0)) }
        return defaults.contains(title) ? name : title
    }
}

extension Moment {
    private var resolvedKind: String? {
        // Cues saved before `kind` existed are recognised by their stored title.
        kind ?? (title.hasPrefix("高抬脚") ? Moment.raisedHeel : title.hasPrefix("快速移动") ? Moment.fastMove : nil)
    }
    var displayTitle: String {
        switch resolvedKind {
        case Moment.raisedHeel: String(localized: "高抬脚 · 候选片段", bundle: AppLanguage.bundle)
        case Moment.fastMove: String(localized: "快速移动 · 回看一下", bundle: AppLanguage.bundle)
        default: title
        }
    }
    var displayDetail: String {
        switch resolvedKind {
        case Moment.raisedHeel: String(localized: "画面中脚跟高于髋部。回看是否为跟勾或高踩；模型无法确认岩点接触。", bundle: AppLanguage.bundle)
        case Moment.fastMove: String(localized: "髋部在画面中的移动较快。可能是动态动作、下落或镜头移动，请结合原片判断。", bundle: AppLanguage.bundle)
        default: detail
        }
    }
}
