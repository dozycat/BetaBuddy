package com.betabuddy.android.pose

/** Which of the 33 landmarks are joined to draw the skeleton. Same list as ios/BetaBuddy/Core/PoseMath.swift. */
object PoseMath {
    val connections: List<Pair<Int, Int>> = listOf(
        11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16, 11 to 23, 12 to 24, 23 to 24, 23 to 25,
        25 to 27, 24 to 26, 26 to 28, 27 to 29, 29 to 31, 27 to 31, 28 to 30, 30 to 32, 28 to 32,
    )
}
