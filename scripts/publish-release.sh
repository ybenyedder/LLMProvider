#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="ybenyedder/LLMProvider"
TAG="${TAG:-v2.0.0}"
AAB_FILE="$ROOT_DIR/LLMProvider-release-2.0.0.aab"
APK_FILE="$ROOT_DIR/app/build/outputs/apk/release/LLMProvider-release-2.0.0.apk"

TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$HOME/.git-credentials" ]]; then
    TOKEN="$(sed -n 's|https://[^:]*:\([^@]*\)@github\.com.*|\1|p' "$HOME/.git-credentials" | head -n 1 || true)"
fi

if [[ -z "$TOKEN" ]]; then
    echo "Usage: GITHUB_TOKEN=ghp_... $0" >&2
    exit 1
fi

for f in "$AAB_FILE" "$APK_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing release artifact: $f (run ./gradlew assembleRelease bundleRelease first)" >&2
        exit 1
    fi
done

echo "=== Verifying GitHub token ==="
curl -sf -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" https://api.github.com/user >/dev/null

echo "=== Pushing main and tag $TAG to origin ==="
cd "$ROOT_DIR"
AUTH_HEADER="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w 0)"
git -c "http.https://github.com/.extraheader=$AUTH_HEADER" push origin main
git -c "http.https://github.com/.extraheader=$AUTH_HEADER" push origin "refs/tags/$TAG" --force

echo "=== Creating GitHub Release $TAG ==="
RELEASE_NOTES="## Tree4Five LLM Provider v2.0.0 — RustingStrata Pure-Rust Edition\n\n- **Pure-Rust Engine (\`libggufllm.so\`):** Complete replacement of \`llama.cpp\` with RustingStrata v2.0 (zero C/C++ dependencies, 16 KiB ELF page alignment).\n- **SIMD Acceleration:** ARM64 NEON (\`ARMv8.0-A\`) & x86_64 AVX2+FMA \`Q8_K\` quantized kernels across 25 GGML formats.\n- **Inference Features:** Batched prompt prefill, prefix KV-cache reuse, N-gram speculative decoding, and lossless \`token_embd\` embedding parity.\n- **Embedded Local HTTP API (\`127.0.0.1:11435\`):** OpenAI (\`/v1/chat/completions\`, \`/v1/embeddings\`, \`/v1/models\`) and Ollama (\`/api/generate\`, \`/api/chat\`, \`/api/tags\`) alongside backward-compatible AIDL.\n- **UI/UX Overhaul:** Dark telemetry dashboard, 6 model presets, atomic \`.part\` GGUF-verified downloads with speed/ETA/cancel, model management, and benchmark playground."

PAYLOAD="$(python3 -c 'import json,sys; print(json.dumps({"tag_name": sys.argv[1], "target_commitish": "main", "name": "v2.0.0 — RustingStrata Pure-Rust Engine", "body": sys.argv[2].replace("\\n", "\n"), "draft": False, "prerelease": False}))' "$TAG" "$RELEASE_NOTES")"

RESP="$(curl -s -w "\n%{http_code}" -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/releases" \
    -d "$PAYLOAD")"
CODE="$(echo "$RESP" | tail -n 1)"
BODY="$(echo "$RESP" | sed '$d')"

if [[ "$CODE" == "422" ]]; then
    echo "Release $TAG already exists, fetching existing release..."
    BODY="$(curl -sf -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/releases/tags/$TAG")"
elif [[ "$CODE" -lt 200 || "$CODE" -ge 300 ]]; then
    echo "Failed to create release (HTTP $CODE): $BODY" >&2
    exit 1
fi

RELEASE_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<< "$BODY")"
UPLOAD_BASE="https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets"

upload_asset() {
    local file="$1"
    local ctype="$2"
    local name
    name="$(basename "$file")"
    echo "Uploading $name..."
    curl -sf -X POST \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: $ctype" \
        --data-binary @"$file" \
        "$UPLOAD_BASE?name=$name" >/dev/null
}

upload_asset "$AAB_FILE" "application/octet-stream"
upload_asset "$APK_FILE" "application/vnd.android.package-archive"

echo "=== Release $TAG published: https://github.com/$REPO/releases/tag/$TAG ==="
