call .venv\Scripts\activate
pyinstaller build.spec --clean --windowed
xcopy /Y "%VIRTUAL_ENV%\Lib\site-packages\tflite_runtime\*.dll" dist\San3a-ML-Runner\
xcopy /Y "%VIRTUAL_ENV%\Lib\site-packages\sounddevice\portaudio_binaries\*.dll" dist\San3a-ML-Runner\
