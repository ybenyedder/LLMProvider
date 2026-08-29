#!/bin/bash
set -e

echo "=== 1. Local Testing ==="
./gradlew testDebugUnitTest

echo "=== 2. Local Deploy (Install on attached device/emulator) ==="
# Optional: if you have a device connected, this will install it
./gradlew installDebug || echo "No device connected, skipping local deploy"

echo "=== 3. Commit and Push ==="
git status
echo "Do you want to stage and commit these changes? (y/n)"
read -r answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    git add .
    echo "Enter commit message:"
    read -r msg
    if [ -z "$msg" ]; then
        msg="Automated pipeline commit"
    fi
    git commit -m "$msg" || true
    echo "Pushing to remote..."
    git push
    echo "Pushed to remote! GitHub Actions will now handle remote testing and deployment."
else
    echo "Skipping commit and push."
fi

echo "=== Pipeline Completed! ==="
