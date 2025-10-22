# Visual Encryption Engine

A real-time cryptographic seed generator that extracts entropy from webcam video feeds. This system analyzes visual data through a grid-based approach, combining camera input with OS-level randomness to produce high-quality cryptographic seeds. Insipired by Cloudflare's Wall of Entropy system in their San Francisco office.

## Overview

The Visual Encryption Engine divides each camera frame into a 16×9 grid (144 cells) and extracts entropy from color statistics, variances, and inter-frame motion deltas. These features are processed through SHA3-256 hashing combined with OS entropy to generate secure 256-bit seeds.

## Features

- **Real-time entropy extraction** from webcam video
- **Dual-mode operation**: Fast mode (color-based) or Advanced mode (includes edge detection & histogram entropy)
- **Quality filtering**: Automatically rejects low-variance frames and low-motion cycles
- **Duplicate detection**: Tracks recent seeds to prevent repetition
- **Live performance monitoring**: FPS counter and entropy quality metrics
- **Grid overlay visualization**: See exactly how frames are divided for analysis
- **OS entropy mixing**: Combines camera entropy with `os.urandom()` for additional security

## Requirements

```bash
pip install opencv-python numpy
```

- Python 3.7+
- Webcam/camera device

## Usage

### Basic Usage

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `g` | Toggle grid overlay display |
| `a` | Toggle advanced entropy features (slower, more entropy) |
| `q` | Quit application |

### Configuration

Edit the constants at the top of `main.py`:

```python
GRID_W = 16              # Grid width (cells)
GRID_H = 9               # Grid height (cells)
FRAMES_PER_CYCLE = 12    # Frames to collect per seed
CYCLE_DELTA_MIN = 0.1    # Minimum motion threshold
FRAME_VAR_MIN = 5.0      # Minimum variance threshold
ENABLE_ADVANCED_ENTROPY = False  # Enable edge/histogram features
SAVE_ENTROPY_DATA = False        # Save raw entropy to file for testing
```

## How It Works

### 1. Frame Capture
- Captures frames from webcam at up to 30 FPS
- Processes frames at standardized 1280×720 resolution
- Displays original aspect ratio to prevent stretching

### 2. Grid Analysis
Each frame is divided into 144 cells (16×9 grid). For each cell, the system extracts:

**Fast Mode (default):**
- Mean color values (B, G, R channels)
- Color variance per channel

**Advanced Mode:**
- All fast mode features
- Edge density (Canny edge detection)
- Histogram entropy

### 3. Motion Detection
- Calculates inter-frame deltas between consecutive frames
- Tracks motion across all cells to ensure sufficient randomness
- Rejects cycles with insufficient motion

### 4. Bit Extraction
- Converts floating-point features to integers (scaled by 2048)
- Extracts 3 LSBs from each value
- Adds microsecond-precision timestamp entropy
- Combines feature bits and delta bits

### 5. Seed Generation
- Collects 12 frames of entropy data per cycle
- Mixes camera entropy with 256 bits of OS entropy (`os.urandom()`)
- Applies SHA3-256 for cryptographic conditioning
- Outputs 64-character hex string (256-bit seed)

## Output Example

```
Process dims: 1280 x 720
Cells are: 80x80
Advanced entropy features: DISABLED (fast mode)
Camera FPS: 30.0

Press 'g' to toggle grid, 'q' to quit, 'a' to toggle advanced entropy

[Performance] FPS: 28.3
Starting cycle... (motion: 15.42, entropy: 0.964)
✓ Seed: 27dac08f189285e1aef756974a4df4d8... (motion=10.96, entropy=0.964)
[Performance] FPS: 29.1
Starting cycle... (motion: 12.33, entropy: 0.957)
✓ Seed: 2ef0ba66b260028c92fa951cb9cad76e... (motion=3.69, entropy=0.960)
```

## Performance

- **Fast Mode**: 25-30 FPS on typical hardware
- **Advanced Mode**: 5-10 FPS (due to edge detection overhead)
- **Entropy Quality**: Typically 0.95-0.97 (out of 1.0 maximum)
- **Seed Generation Rate**: ~1 seed every 12 frames

## Security Considerations

- Seeds are conditioned through SHA3-256 (cryptographically secure hash)
- OS entropy (`os.urandom()`) is mixed with camera data
- Recent seed tracking prevents immediate duplicates
- Motion and variance thresholds ensure sufficient randomness
- Visual entropy combined with system entropy provides defense in depth

## Testing Entropy Quality

Enable entropy data collection for external randomness testing:

```python
SAVE_ENTROPY_DATA = True
```

This saves pre-hash entropy to `entropy_sample.bin`. You can then test with tools like:
- `ent` (entropy analysis)
- `dieharder` (statistical randomness tests)
- NIST Statistical Test Suite

## Troubleshooting

### Low FPS
- Disable advanced entropy features (press `a` or set `ENABLE_ADVANCED_ENTROPY = False`)
- Close other applications using the camera
- Reduce grid resolution

### Dark Image
- Ensure auto-exposure is enabled (default)
- Check camera hardware settings
- Increase ambient lighting

### No Seeds Generated
- Move camera or change scene (needs motion)
- Lower `CYCLE_DELTA_MIN` threshold
- Ensure sufficient lighting (variance check)

## Technical Details

- **Hash Function**: SHA3-256 (Keccak)
- **Grid Size**: 16×9 = 144 cells
- **Cell Size**: 80×80 pixels (at 1280×720 processing resolution)
- **Bits per Frame**: ~3,456 bits (fast mode) or ~4,608 bits (advanced mode)
- **Output Format**: 64-character hexadecimal (256 bits)

## Future Enhancements

- Adaptive thresholds based on environment
- Periodic entropy pool mixing
- Multiple hash algorithm support (Blake2, Argon2)
- Temporal/spatial correlation analysis
- Export seed pool to file
- GUI interface

## License

MIT License - Feel free to use and modify

## Author

Created by ivanbard

---

**Note**: This system is designed for educational and experimental purposes. For production cryptographic applications, consult with security experts and use established entropy sources.