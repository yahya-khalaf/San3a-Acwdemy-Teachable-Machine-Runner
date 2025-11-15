#!/bin/bash

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Removing old build/dist folders ==="
rm -rf build dist

echo "=== Running PyInstaller ==="
pyinstaller build.spec --clean

echo "=== Removing macOS quarantine (important) ==="
xattr -r -d com.apple.quarantine dist/San3a-ML-Runner.app || true

echo "=== Copying required Qt plugins ==="
QTPATH=$(python3 -c "import PySide6; import os; print(os.path.dirname(PySide6.__file__))")

mkdir -p dist/San3a-ML-Runner.app/Contents/PlugIns/platforms
cp -R "$QTPATH/Qt/plugins/platforms" dist/San3a-ML-Runner.app/Contents/PlugIns/

mkdir -p dist/San3a-ML-Runner.app/Contents/PlugIns/imageformats
cp -R "$QTPATH/Qt/plugins/imageformats" dist/San3a-ML-Runner.app/Contents/PlugIns/

echo "=== Build Complete! You can now compress the .app into a ZIP. ==="
