// Kotlin itself comes from the Android Gradle plugin's built-in Kotlin support (AGP 9+);
// the Compose and serialization compiler plugins below pin the Kotlin version.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}
