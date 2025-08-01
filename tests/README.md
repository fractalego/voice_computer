# Voice Computer Tests

This directory contains tests for the Voice Computer system, particularly focusing on the WhisperListener transcription accuracy.

## Test Files

- `test_whisper_listener.py` - Main test file for WhisperListener functionality
- `quick_brown_fox.wav` - Test audio file containing "the quick brown fox jumps over the lazy dog"
- `test_config_with_device.json` - Example configuration file showing device selection
- `run_tests.py` - Test runner script

## Running Tests

### Method 1: Direct execution
```bash
python tests/test_whisper_listener.py
```

### Method 2: Using the test runner
```bash
python tests/run_tests.py
```

### Method 3: Using pytest (if installed)
```bash
pytest tests/ -v
```

## Test Requirements

The tests require the following dependencies:
- `pyaudio` - For audio device handling
- `numpy` - For audio data processing
- `transformers` and `torch` - For Whisper model (if testing transcription)
- `wave` - For WAV file reading (built-in)

## Audio Device Testing

### List Available Devices
Before running tests, you can list available audio devices:
```bash
python run_voice_computer.py --list-devices
```

### Configure Specific Device
To test with a specific microphone device, modify the configuration:
```json
{
  "listener_model": {
    "microphone_device_index": 2
  }
}
```

## Test Structure

### TestWhisperListener
Tests the core WhisperListener functionality:
- `test_transcription_accuracy()` - Tests transcription of the quick brown fox audio file
- `test_empty_audio()` - Tests handling of empty audio data
- `test_silence_audio()` - Tests handling of silent audio

### TestWhisperListenerIntegration  
Tests integration with configuration:
- `test_listener_initialization()` - Tests proper initialization
- `test_config_loading()` - Tests configuration value loading

## Expected Test Behavior

### Successful Transcription Test
The main test loads `quick_brown_fox.wav` and expects to transcribe it as "the quick brown fox jumps over the lazy dog" (or similar with high word similarity).

### Model Dependencies
The test will be skipped if:
- Whisper model cannot be loaded
- Required dependencies are missing
- Audio processing fails due to environment issues

### Similarity Matching
The test uses word-level similarity matching with a 70% threshold, allowing for minor transcription variations while ensuring key words are present.

## Troubleshooting

### ALSA Warnings
ALSA warnings are normal in Linux environments and don't affect test functionality.

### Model Loading Issues
If tests are skipped due to model loading:
1. Ensure transformers and torch are installed
2. Check internet connection for model download
3. Verify sufficient disk space for model files

### Audio Device Issues
If audio device tests fail:
1. Run `--list-devices` to see available devices
2. Check that the configured device index exists
3. Ensure the device supports the required sample rate (16kHz)

## Test Audio File

The `quick_brown_fox.wav` file should contain a clear recording of "the quick brown fox jumps over the lazy dog" at 16kHz sample rate for optimal testing results.