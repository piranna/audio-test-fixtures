# Audio Test Fixtures

Utilities for generating and validating deterministic audio fixtures for
automated testing of audio pipelines.

This project provides:

- A **deterministic audio generator** producing a known sequence of pure tones
- A **validator** that compares a reference WAV against a processed one using
  spectral analysis

It is designed for **end-to-end audio testing**, codec validation, transmission
integrity checks, and CI automation.

## Features

### Audio Generator

- PCM WAV (16-bit, mono, 16 kHz)
- Exactly **10 seconds**
- Ascending chromatic scale covering the **human vocal range**
- Pure sine waves with known frequencies
- Deterministic output (ideal for fixtures)

### Audio Validator

- WAV metadata validation
- Duration and file size comparison
- FFT-based dominant frequency detection per segment
- Frequency tolerance support (lossy codecs)
- Signal-to-Noise Ratio (SNR)
- CI-friendly exit codes

## Installation

### Using Poetry (recommended)

```bash
poetry install
```

### Or as a library (future PyPI)

```bash
pip install audio-test-fixtures
```

## Usage

### 1. Generate a reference fixture

```bash
python -m audio_test_fixtures.generate_vocal_scale
```

Or with a custom filename:

```bash
python -m audio_test_fixtures.generate_vocal_scale my_fixture.wav
```

This produces a 10-second WAV file with known frequencies across the vocal
range.

### 2. Validate a processed audio file

```bash
python -m \
  audio_test_fixtures.validate_audio_transmission reference.wav decoded.wav
```

With custom tolerance and verbose output:

```bash
python -m audio_test_fixtures.validate_audio_transmission \
  reference.wav decoded.wav \
  --tolerance 10.0 \
  --verbose
```

## Validation Metrics

The validator checks:

- WAV format compatibility
- Duration drift
- Dominant frequency per segment
- Frequency accuracy (% within tolerance)
- Mean frequency error
- Signal-to-Noise Ratio (SNR)

### Exit Codes

| Code | Meaning              |
| ---- | -------------------- |
| `0`  | Validation passed    |
| `1`  | Validation failed    |
| `2`  | File or format error |

## Recommended Tolerances

| Scenario            | Frequency Tolerance | Expected Accuracy |
| ------------------- | ------------------- | ----------------- |
| Lossless processing | ±2 Hz               | >95%              |
| Light lossy codec   | ±5 Hz               | >85%              |
| Heavy compression   | ±10 Hz              | >75%              |

## Typical Workflow

```text
Generate fixture
      ↓
Encode / transmit / process
      ↓
Decode output
      ↓
Validate reference vs decoded
```

This makes the tool suitable for:

- Codec validation
- Transport testing (UDP, BLE, RTP, etc.)
- Embedded / mobile audio pipelines
- CI regression tests

## Design Notes

- Uses FFT peak detection instead of waveform comparison
- Robust against amplitude scaling and minor temporal drift
- Focuses on **spectral correctness**, not perceptual metrics

## Dependencies

- Python ≥ 3.8
- NumPy

All dependencies are managed via **Poetry**.

## License

MIT License. See [LICENSE](LICENSE) for details.
