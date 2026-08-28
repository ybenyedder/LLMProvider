# LLM Provider Service

Ceci est l'application Android indépendante qui héberge le modèle GGUF via `llama.cpp` et l'expose aux autres applications via AIDL.

## Fonctionnalités
- Téléchargement du modèle GGUF depuis Hugging Face.
- Lancement du Service LLM en arrière-plan (Foreground Service).
- Interface de test pour envoyer un prompt et recevoir le texte en streaming.

## ⚠️ Important : Compilation Native
Bien que ce projet compile le code Kotlin avec succès en utilisant les classes Java de `de.kherud:llama`, il a besoin de la librairie native (`libjllama.so`) compilée spécifiquement pour Android (architecture ARM) avec le flag Vulkan/CLBlast pour fonctionner **à l'exécution**.

Pour finaliser l'installation de `llama.cpp` :
1. Téléchargez le code source de `java-llama.cpp` et son sous-module `llama.cpp`.
2. Créez un dossier `src/main/cpp` et placez-y le code C++.
3. Dans `app/build.gradle.kts`, décommentez le bloc `externalNativeBuild` et ajoutez les flags CMake :
   `-DGGML_VULKAN=ON` (pour l'accélération GPU Vulkan).
4. Synchronisez Gradle pour compiler l'application avec le NDK Android.
