# LLM Provider (tree4five) 🌲5️⃣

A modern, on-device Large Language Model (LLM) Inference Service for Android.

## Features
- **Local On-Device AI:** Runs GGUF models directly on your Android device using `java-llama.cpp`. No cloud, complete privacy.
- **AIDL Service Architecture:** Exposes a persistent Foreground Service, allowing any other app on the device to bind to it and generate text seamlessly.
- **Dynamic Model Loading:** Fetch models directly via URL (HTTP/HTTPS) or load local `.gguf` files using the Android Storage Access Framework.
- **Material 3 Design:** A beautiful, responsive interface utilizing the custom tree4five color palette.
- **Android 14 Ready:** Fully compliant with Android 14 Foreground Service (FGS) requirements (`FOREGROUND_SERVICE_SPECIAL_USE`).

## How to Use
1. **Download the App:** Build the project or grab the latest APK.
2. **Setup a Model:** Launch the app, enter a HuggingFace GGUF URL (e.g., SmolLM) or select a local model file, and tap "Fetch URL" or "Select File".
3. **Start the Service:** Tap "Start Service" to initialize the inference engine in the background.
4. **Test Inference:** Enter a prompt in the test box and watch the model stream the response in real-time!

## Build Instructions
This project uses Gradle. To build the debug APK:
```bash
./gradlew assembleDebug
```
To run the automated tests:
```bash
./gradlew testDebugUnitTest
./gradlew connectedAndroidTest
```

## Integrating into your app
Since this runs as an AIDL service, you can bind to `com.example.gguf.ACTION_LLM_SERVICE` from your own apps and pass the `ILLMCallback` stub to receive token streams asynchronously!

## License
MIT
