plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.tree4five.gguf"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.tree4five.gguf"
        minSdk = 26
        targetSdk = 36
        versionCode = 15
        versionName = "1.1.6"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // The native bridge is built only for arm64: the target device
            // (SM-P610) is arm64-v8a and each extra ABI doubles the .so size.
            // x86_64 is added to debug builds only so the emulator can run
            // the instrumentation suite.
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("../release.keystore")
            storePassword = project.findProperty("MYAPP_RELEASE_STORE_PASSWORD") as String? ?: "password123"
            keyAlias = project.findProperty("MYAPP_RELEASE_KEY_ALIAS") as String? ?: "key0"
            keyPassword = project.findProperty("MYAPP_RELEASE_KEY_PASSWORD") as String? ?: "password123"
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
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // llama.cpp is now built from source in src/main/cpp (libggufllm.so) and
    // accessed through our own JNI bridge (LlmNative). The de.kherud:llama
    // binding was removed: it shipped a second libllama that must never be
    // loaded in the same process as ours.

    testImplementation("org.json:json:20230227")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
}
