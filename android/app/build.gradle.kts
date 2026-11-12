import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

// Release signing is read from android/keystore.properties (ignored by Git) or from the
// environment. Neither the keystore nor its passwords belong in the repository.
//   storeFile=/absolute/or/relative/to/android/path.jks
//   storePassword=…   keyAlias=…   keyPassword=…
val keystoreProperties = Properties().apply {
    val file = rootProject.file("keystore.properties")
    if (file.isFile) file.inputStream().use { load(it) }
}

fun signingValue(property: String, environment: String): String? =
    (keystoreProperties.getProperty(property) ?: System.getenv(environment))?.takeIf { it.isNotBlank() }

android {
    namespace = "com.betabuddy.android"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.betabuddy.android"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    val releaseStore = signingValue("storeFile", "BETABUDDY_KEYSTORE_FILE")
    val releaseSigning = if (releaseStore == null) null else signingConfigs.create("release") {
        storeFile = rootProject.file(releaseStore)
        storePassword = signingValue("storePassword", "BETABUDDY_KEYSTORE_PASSWORD")
        keyAlias = signingValue("keyAlias", "BETABUDDY_KEY_ALIAS")
        keyPassword = signingValue("keyPassword", "BETABUDDY_KEY_PASSWORD")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Without a configured keystore the release build is signed with the debug key,
            // which is enough to install and test it but not to publish it.
            signingConfig = releaseSigning ?: signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.webkit)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.material3)

    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)

    // Logbook: video playback and on-device pose estimation.
    implementation(libs.androidx.media3.exoplayer)
    implementation(libs.mediapipe.tasks.vision)

    // Unit tests are plain JVM tests: no Android classes, no emulated framework.
    testImplementation(libs.junit)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
}
