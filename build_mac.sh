#!/bin/bash

# Stop on first error
set -e

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Removing old build/dist folders ==="
rm -rf build dist

echo "=== Running PyInstaller ==="
pyinstaller build.spec --clean --windowed

echo "=== Fixing macOS quarantine attribute ==="
# Remove Apple “quarantine” flag, otherwise the app cannot access camera/mic
xattr -r -d com.apple.quarantine "dist/San3a-ML-Runner.app" || true

echo "=== Ensuring Qt plugins are properly bundled ==="
# Sometimes QtWebEngine or platform plugins are missing—force copy
QT_PATH=$(python3 -c "import PySide6, os; print(os.path.dirname(PySide6.__file__))")

mkdir -p "dist/San3a-ML-Runner.app/Contents/Plugins"
cp -R "$QT_PATH/Qt/plugins/" "dist/San3a-ML-Runner.app/Contents/Plugins/" || true

echo "=== Ensuring Qt frameworks are bundled ==="
mkdir -p "dist/San3a-ML-Runner.app/Contents/Frameworks"
cp -R "$QT_PATH/Qt/lib/" "dist/San3a-ML-Runner.app/Contents/Frameworks/" || true

echo "=== macOS build completed successfully ==="
echo "Your app is at: dist/San3a-ML-Runner.app"
