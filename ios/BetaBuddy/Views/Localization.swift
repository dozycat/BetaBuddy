import SwiftUI

// Display text for values that are stored language-independently or in the source language.

/// Stored sentinel for a climb without a grade. Kept as is so existing archives stay readable.
let ungradedValue = "未标级"

func gradeText(_ grade: String) -> String {
    grade == ungradedValue ? String(localized: "未标级") : grade
}

func percentText(_ fraction: Double) -> String {
    fraction.formatted(.percent.precision(.fractionLength(0)))
}

extension Moment {
    private var resolvedKind: String? {
        // Cues saved before `kind` existed are recognised by their stored title.
        kind ?? (title.hasPrefix("高抬脚") ? Moment.raisedHeel : title.hasPrefix("快速移动") ? Moment.fastMove : nil)
    }
    var displayTitle: String {
        switch resolvedKind {
        case Moment.raisedHeel: String(localized: "高抬脚 · 候选片段")
        case Moment.fastMove: String(localized: "快速移动 · 回看一下")
        default: title
        }
    }
    var displayDetail: String {
        switch resolvedKind {
        case Moment.raisedHeel: String(localized: "画面中脚跟高于髋部。回看是否为跟勾或高踩；模型无法确认岩点接触。")
        case Moment.fastMove: String(localized: "髋部在画面中的移动较快。可能是动态动作、下落或镜头移动，请结合原片判断。")
        default: detail
        }
    }
}
