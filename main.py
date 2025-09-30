import cv2
import numpy as np
import hashlib, time, os
from collections import deque

# using default camera 
cam = cv2.VideoCapture(0)

# Set camera properties for stability and consistent quality
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure for consistency
cam.set(cv2.CAP_PROP_EXPOSURE, -6)  # Set manual exposure
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cam.set(cv2.CAP_PROP_AUTO_WB, 1)  # Keep auto white balance for color diversity
cam.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Set consistent brightness
cam.set(cv2.CAP_PROP_CONTRAST, 128)  # Set consistent contrast

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# variables for the input grid to simplify calculations
GRID_W = 16
GRID_H = 9

# resize input frame to 1280x720 to standardize
PROC_W = 1280
PROC_H = 720

# capture 12 back-to-back frames, 2.4 s per cycle, one cycle per min
# use the collected frames to concatenate raw bytes and hash into a seed
FRAMES_PER_CYCLE = 12 
FRAME_GAP_S = 0.2
CYCLES_PERIOD_S = 60.0

FRAME_VAR_MIN = 5.0 # global variance below this means a bad frame (too similar colors)
CYCLE_DELTA_MIN = 0.5 # mean delta per channel lower than this means bad cycle if shared between frames
RECENT_SEEDS = deque(maxlen=50)
SAVE_ENTROPY_DATA = False  # Set to True to save raw entropy for testing

show_grid = True
prev_stats = None

cycle_buf = bytearray()
frames_this_cycle = 0
last_cycle_time = time.time()
delta_history = []  # Move this outside the loop

def prep_frame(frame):
    return cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

# the dimensions of the cells within the simplified grid
def cell_dimensions():
    CELL_W = PROC_W // GRID_W
    CELL_H = PROC_H // GRID_H
    return CELL_W, CELL_H

# grid creation to showcase cells
def draw_grid(img, color=(0,255,0), thickness=2):
    h, w = img.shape[:2]
    cw, ch = cell_dimensions()

    for gx in range(1, GRID_W):
        x = gx * cw
        cv2.line(img, (x, 0), (x, h), color, thickness)
    # horizontal lines
    for gy in range(1, GRID_H):
        y = gy * ch
        cv2.line(img, (0, y), (w, y), color, thickness)
    
    return img

# cell content extraction with enhanced entropy
def extract_cell_strats(frame):
    h, w = frame.shape[:2]
    cw, ch = cell_dimensions()
    stats = np.zeros((GRID_H*GRID_W, 8), dtype=np.float32)  # Extended to include more features
    idx = 0

    for gy in range(GRID_H):
        y0, y1 = gy*ch, (gy+1)*ch
        for gx in range(GRID_W):
            x0, x1 = gx*cw, (gx+1)*cw
            cell = frame[y0:y1, x0:x1]

            # means/vars per channel
            means = cell.mean(axis=(0,1)) # B, G, R
            vars_ = cell.var(axis=(0,1))
            
            # Additional entropy sources
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            edge_density = cv2.Canny(gray_cell, 50, 150).mean()  # Edge information
            
            # Histogram entropy (simplified)
            hist = cv2.calcHist([gray_cell], [0], None, [16], [0, 256])
            hist_entropy = -np.sum(hist * np.log2(hist + 1e-10)) / hist.sum()
            
            stats[idx, 0:3] = means
            stats[idx, 3:6] = vars_
            stats[idx, 6] = edge_density
            stats[idx, 7] = hist_entropy
            idx += 1
    return stats

def compute_mean_deltas(curr_stats, prev_stats):
    if prev_stats is None:
        return np.zeros((GRID_H*GRID_W, 8), dtype=np.float32)  # Updated for extended features
    return np.abs(curr_stats - prev_stats)

def lsb_bits_from_int(x, k):
    #return list of k LSBs
    return [(x >> i) & 1 for i in range(k)]

def extract_bits_from_feats(stats, mean_deltas, mean_lsb=3, delta_lsb=3):
    # Enhanced bit extraction with more entropy sources
    bits = []
    # Increased scale factors for better distribution
    M_SCALE = 2048
    D_SCALE = 2048

    # Extract from all features, not just means
    for feature_idx in range(stats.shape[1]):
        vals = stats[:, feature_idx]
        val_ints = (vals * M_SCALE).astype(np.int64)
        for val in val_ints:
            bits.extend(lsb_bits_from_int(int(val), mean_lsb))

    # Extract from all delta features
    for feature_idx in range(mean_deltas.shape[1]):
        deltas = mean_deltas[:, feature_idx]
        delta_ints = (deltas * D_SCALE).astype(np.int64)
        for val in delta_ints:
            bits.extend(lsb_bits_from_int(int(val), delta_lsb))
    
    # Add timestamp entropy (microsecond precision)
    timestamp_bits = lsb_bits_from_int(int(time.time() * 1000000) % (2**32), 8)
    bits.extend(timestamp_bits)
    
    return bits

def bits_to_bytes(bits):
    out = bytearray()
    byte = 0
    count = 0
    for b in bits:
        byte = (byte << 1) | (b&1)
        count += 1
        if count == 8:
            out.append(byte)
            byte, count = 0, 0
    if count > 0: #left overs padded with 0s
        out.append(byte << (8 - count))
    return bytes(out)

def start_cycle():
    return bytearray(), 0

def add_frame_to_cycle(cycle_buf, raw_bytes):
    cycle_buf.extend(raw_bytes)
    return cycle_buf

def finish_cycle(cycle_buf):
    # Mix with OS entropy for additional randomness
    os_entropy = os.urandom(32)  # 256 bits of OS entropy
    
    # Combine camera entropy with OS entropy
    combined_entropy = cycle_buf + os_entropy
    
    # Use SHA3-256 for conditioning (cryptographically secure)
    seed = hashlib.sha3_256(combined_entropy).digest()
    return seed

def frame_passes_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = gray.var()
    return v >= FRAME_VAR_MIN, v

def cycle_passes_motion(delta_history):
    if not delta_history: return False, 0.0
    avg = sum(delta_history) / len(delta_history)
    return avg >= CYCLE_DELTA_MIN, avg

def assess_entropy_quality(raw_bytes):
    """Simple entropy assessment for the raw bytes"""
    if len(raw_bytes) == 0:
        return 0.0
    
    # Calculate Shannon entropy
    byte_counts = np.bincount(raw_bytes, minlength=256)
    probabilities = byte_counts / len(raw_bytes)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Normalize to 0-1 scale (max entropy for uniform distribution is 8 bits)
    return entropy / 8.0

# Optional: Save raw entropy data for external randomness testing
def save_entropy_sample(cycle_buf, filename="entropy_sample.bin"):
    """Save pre-hash bytes for randomness testing"""
    with open(filename, "ab") as f:
        f.write(cycle_buf)

def seed_is_new(seed):
    hx = seed.hex()
    if hx in RECENT_SEEDS:
        return False
    RECENT_SEEDS.append(hx)
    return True

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

cw, ch = cell_dimensions()
print(f"Process dims: {PROC_W} x {PROC_H}")
print(f"Cells are: {cw}x{ch}")

#run a loop until input to close
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to read frame")
        break
        
    # Resize frame for consistent processing
    frame = prep_frame(frame)
    
    out.write(frame)
    if show_grid:
        draw_grid(frame)
    cv2.imshow('Webcam', frame)

    # per frame: drop bad frames early
    ok_var, var_val = frame_passes_variance(frame)
    if not ok_var:
        print(f"[WARN] Low variance frame (var={var_val:.2f}); skipping.")
        continue

    # inter-frame deltas to increase entropy through motion
    stats = extract_cell_strats(frame)
    mean_deltas = compute_mean_deltas(stats, prev_stats)
    prev_stats = stats

    # bits to bytes
    raw_bits = extract_bits_from_feats(stats, mean_deltas, mean_lsb=3, delta_lsb=3)
    raw_bytes = bits_to_bytes(raw_bits)
    
    # Assess entropy quality
    entropy_quality = assess_entropy_quality(raw_bytes)
    
    # Track motion for cycle quality assessment
    delta_scalar = float(mean_deltas.mean())
    delta_history.append(delta_scalar)

    #cycle in-loop logic
    if frames_this_cycle == 0:
        cycle_buf, frames_this_cycle = start_cycle()

    cycle_buf = add_frame_to_cycle(cycle_buf, raw_bytes)
    frames_this_cycle += 1

    # Check if cycle is complete
    if frames_this_cycle >= FRAMES_PER_CYCLE:
        # Assess cycle quality before generating seed
        ok_motion, avg_delta = cycle_passes_motion(delta_history)
        
        if ok_motion:
            # Optionally save entropy data for testing
            if SAVE_ENTROPY_DATA:
                save_entropy_sample(cycle_buf)
                
            seed = finish_cycle(cycle_buf)
            if seed_is_new(seed):
                print(f"✓ Seed generated (motion={avg_delta:.3f}, entropy={entropy_quality:.3f}): {seed.hex()}")
            else:
                print("[WARN] Seed repeated; discarding.")
        else:
            print(f"[WARN] Low motion/entropy this cycle (avg delta {avg_delta:.3f}); discarding.")
        
        # Reset for next cycle
        cycle_buf, frames_this_cycle = start_cycle()
        delta_history = []
        last_cycle_time = time.time()

    # Display diagnostics periodically
    if frames_this_cycle % 5 == 0:  # Every 5 frames within a cycle
        print(f"Frame {frames_this_cycle}/{FRAMES_PER_CYCLE}, Motion: {delta_scalar:.3f}, Entropy: {entropy_quality:.3f}")

    if cv2.waitKey(1) == ord('g'): # this button press is iffy, sometimes requires double presses
        show_grid = not show_grid
    elif cv2.waitKey(1) == ord('q'): #press q to exit
        break

# release capture and writers
cam.release()
out.release()
cv2.destroyAllWindows()

print(f"\nSession complete. Generated {len(RECENT_SEEDS)} unique seeds.")

# Additional improvements to consider:
# X Mix with OS randomness 
# X Enhanced entropy extraction (edge density, histogram entropy)
# X Better motion detection and frame quality assessment
# X Improved bit extraction with more LSBs and higher scaling
# X Real-time entropy quality monitoring
# - Adaptive thresholds based on environment
# - Periodic entropy pool mixing
# - Multiple hash algorithms (Blake2, Argon2)
# - Temporal correlation analysis
# - Spatial correlation analysis within grid cells