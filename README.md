# Universal Teachable Machine Controller

A comprehensive desktop application for real-time inference with Google's Teachable Machine models. Supports both image (webcam) and audio (microphone) models with hardware control via serial communication.

## Features

### 🎯 Multi-Model Support
- **Image Models**: Real-time webcam feed processing through image classification models
- **Audio Models**: Real-time microphone input processing through audio classification models
- **Automatic Model Type Detection**: Analyzes input tensor shape to determine model type

### �� Real-time Inference Pipeline
- **Camera Thread**: Continuous frame capture with platform-specific backends (AVFoundation on macOS)
- **Audio Recorder Thread**: Real-time audio chunk processing
- **Inference Workers**: Separate threads for model prediction to prevent UI freezing
- **Frame/Audio Queues**: Buffered data flow between capture and inference

### 🔌 Hardware Integration
- **Serial Communication**: Controls external devices (Arduino, robots, etc.)
- **Label Mapping**: Maps classification results to serial commands
- **Configurable Thresholds**: Confidence-based filtering for reliable control

### 🖥️ User Interface
- **Model Management**: Load `.tflite` models and associated labels
- **Real-time Preview**: Live camera feed or audio status display
- **Label Configuration**: Table-based mapping of labels to serial characters
- **Status Monitoring**: Visual indicators for application state
- **Live Deployment Mode**: Serial output with confidence thresholding

## Installation

### Prerequisites
- Python 3.8 or higher
- macOS (with proper camera/microphone permissions)
- Webcam (for image models)
- Microphone (for audio models)
- Serial device (optional, for hardware control)

### Dependencies
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### macOS Specific Setup
1. **Camera/Microphone Permissions**: Grant permissions in System Preferences > Security & Privacy > Privacy
2. **Audio System**: Ensure PortAudio is installed (usually comes with sounddevice)

## Usage

### 1. Launch the Application
```bash
python app.py
```

### Building macOS app
To build a macOS app bundle using PyInstaller (recommended quick path), follow these steps:

1. Create and activate a virtualenv, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pyinstaller
```

2. Run the helper build script:

```bash
./build/build_mac.sh
```

See `build/README.md` for more details and troubleshooting tips (Qt plugins, PortAudio, codesigning).


### 2. Load a Model
1. Click "Load Model" button
2. Select your `.tflite` file
3. The application will automatically detect the model type and load associated labels

### 3. Configure Label Mapping
1. The application automatically populates the label mapping table
2. Edit the "Serial Command" column to map each label to a character/command
3. Add or remove mappings as needed

### 4. Set Confidence Threshold
1. Use the confidence threshold slider (0-100%)
2. Only predictions above this threshold will trigger serial commands

### 5. Connect Hardware (Optional)
1. Connect your serial device (Arduino, etc.)
2. Click "Refresh" to scan for available ports
3. Select your device and click "Connect"

### 6. Start Inference
1. Click "Start Inference" to begin real-time classification
2. Monitor the live preview and prediction results
3. Serial commands will be sent automatically when confidence threshold is met

## Model File Structure

### Required Files
```
model_directory/
├── model.tflite     # TensorFlow Lite model
└── labels.txt       # Class labels (one per line)
```

### Label File Format
```
class_0_name
class_1_name  
class_2_name
```

## Supported Model Types

### Image Models
- Input: Webcam feed
- Output: Classification predictions
- Common formats: 224x224x3, 299x299x3, etc.

### Audio Models
- Input: Microphone audio
- Output: Classification predictions
- Common formats: 1D audio arrays, spectrograms

## Serial Communication

### Protocol
- **Data Format**: UTF-8 encoded strings
- **Default Baud Rate**: 115200
- **Commands**: Single character commands by default (configurable)

### Example Arduino Code
```cpp
void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    
    switch(command) {
      case 'A':
        // Handle class A
        digitalWrite(LED_PIN, HIGH);
        break;
      case 'B':
        // Handle class B
        digitalWrite(LED_PIN, LOW);
        break;
      // Add more cases as needed
    }
  }
}
```

## Application States

- **Idle**: No model loaded
- **Ready**: Model loaded, ready for inference
- **Running**: Active inference session
- **Serial Connected**: Hardware communication established
- **Error**: Operation failed

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Try different camera backends (AVFoundation, DirectShow, V4L2)
- Check if camera is being used by another application

### Audio Issues
- Ensure microphone permissions are granted
- Check audio device availability
- Verify PortAudio installation

### Model Loading Issues
- Verify `.tflite` file is valid
- Check if labels.txt exists in the same directory
- Ensure model input shape is supported

### Serial Communication Issues
- Verify correct port selection
- Check baud rate compatibility
- Ensure device is not in use by another application

## Performance Optimization

### Resource Management
- Limited queue sizes prevent memory buildup
- Frame skipping when inference can't keep pace
- Thread-safe model access for concurrent inference

### Inference Optimization
- Batch processing preparation
- Tensor shape validation and auto-correction
- Type casting for model compatibility

## Extension Points

### Potential Enhancements
- Support for custom pre-processing pipelines
- Multiple model simultaneous execution
- Data logging and export capabilities
- Custom serial protocols
- Network streaming output

## Technical Details

### Thread Architecture
- **Main Thread**: GUI and user interaction
- **Camera Thread**: Video capture with fallback backends
- **Audio Thread**: Real-time audio streaming
- **Inference Workers**: Model prediction processing
- **Serial Worker**: Hardware communication

### Error Handling
- Graceful thread termination
- Queue cleanup on shutdown
- Resource release guarantees
- Comprehensive error reporting

## License

This project is open source. Please check the license file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the application logs
3. Submit an issue with detailed information about your setup
