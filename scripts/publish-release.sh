#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="ybenyedder/LLMProvider"
TAG="${1:-${TAG:-v2.4.1}}"
VERSION="${TAG#v}"
AAB_FILE="$ROOT_DIR/LLMProvider-release-${VERSION}.aab"
APK_FILE="$ROOT_DIR/app/build/outputs/apk/release/LLMProvider-release-${VERSION}.apk"

TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$HOME/.git-credentials" ]]; then
    TOKEN="$(sed -n 's|https://[^:]*:\([^@]*\)@github\.com.*|\1|p' "$HOME/.git-credentials" | head -n 1 || true)"
fi

if [[ -z "$TOKEN" ]]; then
    echo "Usage: GITHUB_TOKEN=ghp_... $0 [tag]" >&2
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
RELEASE_NOTES="## Tree4Five LLM Provider ${TAG} — Clean Chat UI + RustingStrata Engine\n\n- **Rikka-Style Chat Interface:** Clean deep slate & emerald palette (\`#0B0F17\`, \`#131A26\`, \`#10B981\`, \`#22D3EE\`), top model selector pill, multi-turn conversation bubbles, and floating bottom composer with IME inset support.\n- **3-Tab Bottom Navigation (\`Chat\`, \`Modèles\`, \`Moteur\`):** One-tap installed GGUF switcher, quick HuggingFace catalogue rows, live engine telemetry (\`tok/s\`, RSS, KV cache, N-gram acceptance), and local HTTP API (\`:11435\`).\n- **Pure-Rust Inference (\`libggufllm.so\`):** RustingStrata SIMD kernels (ARM64 NEON & x86_64 AVX2+FMA) with 16 KiB ELF alignment, batched prefill, prefix KV-cache reuse, and N-gram speculative decoding."

PAYLOAD="$(python3 -c 'import json,sys; print(json.dumps({"tag_name": sys.argv[1], "target_commitish": "main", "name": f"{sys.argv[1]} — Clean Chat UI + RustingStrata", "body": sys.argv[2].replace("\\n", "\n"), "draft": False, "prerelease": False}))' "$TAG" "$RELEASE_NOTES")"

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
