# Tree4Five LLM Provider 🌲5️⃣

**Tree4Five LLM Provider** is a local LLM server designed specifically for Android. It acts as an on-device AI backend for *other* Android applications that want to run Large Language Models (LLMs) locally without relying on cloud APIs. By exposing a unified AIDL service, it allows any client application on your device to seamlessly generate text and access powerful AI capabilities completely offline and with absolute privacy.

## Features
- **Local On-Device AI:** Runs GGUF models directly on your Android device using `java-llama.cpp`. No cloud, complete privacy.
- **AIDL Service Architecture:** Exposes a persistent Foreground Service, allowing any other app on the device to bind to it and generate text seamlessly.
- **Dynamic Model Loading:** Fetch models directly via URL (HTTP/HTTPS) or load local `.gguf` files using the Android Storage Access Framework.
- **Material 3 Design:** A beautiful, responsive interface utilizing the custom tree4five color palette.
- **Android 14 Ready:** Fully compliant with Android 14 Foreground Service (FGS) requirements (`FOREGROUND_SERVICE_SPECIAL_USE`).

## How to Use
1. **Download the App:** Build the project or grab the latest APK.
2. **Setup a Model:** Launch the app, enter a HuggingFace GGUF URL (e.g., Qwen) or select a local model file, and tap "Fetch URL" or "Select File".
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

## Integrating into your app (Client Implementation)
Since this runs as an AIDL service, any Android app can bind to it. Here is how you can use it in your own client application (e.g., your GitHub projects).

**Quick Example:**
Here is a simple example of how to bind the service, send a prompt, and print the real-time response:

```kotlin
// 1. Bind to the service
val intent = Intent("com.tree4five.gguf.ACTION_LLM_SERVICE")
intent.setPackage("com.tree4five.gguf")
bindService(intent, connection, Context.BIND_AUTO_CREATE)

// 2. Send a prompt & print the response
llmService.generateTextStream("Explain Quantum Physics", object : ILLMCallback.Stub() {
    override fun onTokenReceived(token: String) {
        print(token) // Streams words in real-time!
    }
    
    override fun onGenerationComplete(fullText: String) {
        println("\nDone!")
    }
})
```
*Visit our GitHub repository for the full AIDL interfaces and detailed setup guides!*

### 1. Add the AIDL files to your client app
In your client app, create the exact same AIDL interfaces under `app/src/main/aidl/com/tree4five/gguf/`:

**ILLMCallback.aidl**
```aidl
package com.tree4five.gguf;
interface ILLMCallback {
    oneway void onTokenReceived(String token);
    oneway void onGenerationComplete(String fullText);
}
```

**ILLMService.aidl**
```aidl
package com.tree4five.gguf;
import com.tree4five.gguf.ILLMCallback;
interface ILLMService {
    oneway void generateTextStream(String prompt, ILLMCallback callback);
    oneway void stopGeneration();
    String getVersion();
}
```

### 2. Bind to the Service (Kotlin Coroutines Example)
Here is an example service wrapper you can use in your client app to call the LLM easily using Kotlin Coroutines:

```kotlin
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import com.tree4five.gguf.ILLMCallback
import com.tree4five.gguf.ILLMService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class LLMClient(private val context: Context) {
    private var llmService: ILLMService? = null
    private var isBound = false

    private suspend fun getService(): ILLMService = suspendCancellableCoroutine { continuation ->
        if (llmService != null) {
            continuation.resume(llmService!!)
            return@suspendCancellableCoroutine
        }

        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                llmService = ILLMService.Stub.asInterface(service)
                isBound = true
                continuation.resume(llmService!!)
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                llmService = null
                isBound = false
            }
        }

        val intent = Intent("com.tree4five.gguf.ACTION_LLM_SERVICE").apply {
            setPackage("com.tree4five.gguf") // Target the LLM Provider app
        }
        
        val bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        if (!bound) {
            continuation.resumeWithException(Exception("Impossible de se lier au service LLM provider"))
        }

        continuation.invokeOnCancellation {
            if (isBound) {
                context.unbindService(connection)
                isBound = false
                llmService = null
            }
        }
    }

    suspend fun generateText(prompt: String): String = withContext(Dispatchers.IO) {
        val service = getService()
        suspendCancellableCoroutine { continuation ->
            val callback = object : ILLMCallback.Stub() {
                override fun onTokenReceived(token: String) {
                    // You can emit tokens here if you want real-time streaming
                }
                
                override fun onGenerationComplete(fullText: String) {
                    continuation.resume(fullText)
                }
            }
            service.generateTextStream(prompt, callback)
        }
    }
}
```

### 3. Using "Tools" (Structured JSON Output)
Since the local GGUF models are running purely text-to-text, you can simulate "tools" or API responses by instructing the model to reply strictly in JSON format.

Here is how another app (like NutriMate) calls the LLM to return a structured data object instead of raw text:

1. **Prompt Engineering:** Tell the model to reply in pure JSON.
2. **Clean the Output:** Remove any markdown blocks (like ` ```json `).
3. **Parse with GSON:** Deserialize the JSON into your Kotlin Data Class.

```kotlin
import com.google.gson.GsonBuilder
import android.util.Log

data class Meal(
    val titre: String,
    val calories: Int,
    val ingredients: List<Ingredient>
)

data class Ingredient(val nom: String, val quantite: String)

suspend fun generateStructuredMeal(client: LLMClient, type: String): Meal? {
    val prompt = """
        Write a tasty $type recipe.
        You MUST reply in pure JSON format only, with the exact following keys:
        - "titre" (string)
        - "calories" (integer)
        - "ingredients" (array of objects with "nom" and "quantite" strings)
        Do not add any other text or explanation.
    """.trimIndent()
    
    val resultJson = client.generateText(prompt)
    
    // Clean up Markdown formatting from the model output
    var cleanJson = resultJson.trim()
    if (cleanJson.contains("```json")) cleanJson = cleanJson.substringAfter("```json")
    else if (cleanJson.contains("```")) cleanJson = cleanJson.substringAfter("```")
    if (cleanJson.contains("```")) cleanJson = cleanJson.substringBeforeLast("```")
    cleanJson = cleanJson.trim()
    
    return try {
        val gson = GsonBuilder().setLenient().create()
        gson.fromJson(cleanJson, Meal::class.java)
    } catch (e: Exception) {
        Log.e("LLMClient", "Failed to parse JSON: $cleanJson", e)
        null
    }
}
```

## License
MIT
