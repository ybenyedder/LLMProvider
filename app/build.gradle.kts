plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val buildRustNative = tasks.register<Exec>("buildRustNative") {
    workingDir = rootProject.projectDir
    val sdkPath = android.sdkDirectory.absolutePath
    environment("ANDROID_SDK_ROOT", sdkPath)
    environment("ANDROID_HOME", sdkPath)
    commandLine("bash", "${rootProject.projectDir}/scripts/build-rust-android.sh")
}

tasks.named("preBuild").configure {
    dependsOn(buildRustNative)
}

android {
    namespace = "com.tree4five.gguf"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.tree4five.gguf"
        minSdk = 26
        targetSdk = 36
        versionCode = 25
        versionName = "2.4.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // Release builds target arm64-v8a; debug adds x86_64 for emulators.
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("../release.keystore")
            storePassword = project.findProperty("MYAPP_RELEASE_STORE_PASSWORD") as String? ?: "password"
            keyAlias = project.findProperty("MYAPP_RELEASE_KEY_ALIAS") as String? ?: "release"
            keyPassword = project.findProperty("MYAPP_RELEASE_KEY_PASSWORD") as String? ?: "password"
        }
    }

    buildTypes {
        debug {
            ndk {
                abiFilters += "x86_64"
            }
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        buildConfig = true
        viewBinding = true
        aidl = true
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.all {
            it.systemProperty(
                "java.library.path",
                file("src/main/rust/ggufllm/target/release").absolutePath
            )
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Pure-Rust inference engine (RustingStrata v2.0) built from app/src/main/rust/ggufllm
    // into libggufllm.so and accessed through LlmNative JNI.

    testImplementation("org.json:json:20230227")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
}
