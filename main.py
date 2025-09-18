import cv2
import numpy as np
import hashlib, time
from collections import deque

# using default camera 
cam = cv2.VideoCapture(0)

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

show_grid = True
prev_stats = None

cycle_buf = bytearray()
frames_this_cycle = 0
last_cycle_time = time.time()

def prep_frame(frame):
    return cv2.resizeWindow(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

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

# cell content extraction
def extract_cell_strats(frame):
    h, w = frame.shape[:2]
    cw, ch = cell_dimensions()
    stats = np.zeros((GRID_H*GRID_W, 6), dtype=np.float32)
    idx = 0

    for gy in range(GRID_H):
        y0, y1 = gy*ch, (gy+1)*ch
        for gx in range(GRID_W):
            x0, x1 = gx*cw, (gx+1)*cw
            cell = frame[y0:y1, x0:x1]

            # means/vars per channel
            means = cell.mean(axis=(0,1)) # B, G, R
            vars_ = cell.var(axis=(0,1))
            stats[idx, 0:3] = means
            stats[idx, 3:6] = vars_
            idx += 1
    return stats

def compute_mean_deltas(curr_stats, prev_stats):
    if prev_stats is None:
        return np.zeros((GRID_H*GRID_W, 3), dtype=np.float32)
    return np.abs(curr_stats[:, 0:3] - prev_stats[:, 0:3])

def lsb_bits_from_int(x, k):
    #return list of k LSBs
    return [(x >> i) & 1 for i in range(k)]

def extract_bits_from_feats(stats, mean_deltas, mean_lsb=2, delta_lsb=2):
    # stats: (Ncells, 6) floats; mean_deltas: (Ncells, 3) floats
    bits = []
    # scale factors tuned to spread vals over ints
    M_SCALE = 1024
    D_SCALE = 1024

    means = stats[:, 0:3]
    deltas = mean_deltas

    #means to ints to LSBs
    m_int = (means * M_SCALE).astype(np.int64)
    for val in m_int.flatten():
        bits.extend(lsb_bits_from_int(int(val), mean_lsb))

    #deltas to ints to LSBs
    d_int = (deltas * D_SCALE).astype(np.int64)
    for val in d_int.flatten():
        bits.extend(lsb_bits_from_int(int(val), delta_lsb))
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
    #SHA3-256 econditioning for seed
    seed = hashlib.sha3_256(cycle_buf).digest()
    return seed

def frame_passes_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = gray.var()
    return v >= FRAME_VAR_MIN, v

def cycle_passes_motion(delta_history):
    if not delta_history: return False, 0.0
    avg = sum(delta_history) / len(delta_history)
    return avg >= CYCLE_DELTA_MIN, avg

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
    out.write(frame)
    if show_grid:
        draw_grid(frame)
    cv2.imshow('Webcam', frame)

    # inter-frame deltas to increase entropy through motion
    stats = extract_cell_strats(frame)
    mean_deltas = compute_mean_deltas(stats, prev_stats)
    prev_stats = stats

    # bits to bytes
    raw_bits = extract_bits_from_feats(stats, mean_deltas, mean_lsb=2, delta_lsb=2)
    raw_bytes = bits_to_bytes(raw_bits)

    #cycle in-loop logic
    if frames_this_cycle == 0:
        cycle_buf, frames_this_cycle = start_cycle()

    cycle_buf = add_frame_to_cycle(cycle_buf, raw_bytes)
    frames_this_cycle += 1

    time.sleep(FRAME_GAP_S)

    if frames_this_cycle >= FRAMES_PER_CYCLE:
        seed = finish_cycle(cycle_buf)
        print("Seed (hex): ", seed.hex())
        # reset timing to stick to the one cycle per min
        sleep_left = max(0.0, CYCLES_PERIOD_S - FRAMES_PER_CYCLE*FRAME_GAP_S)
        time.sleep(sleep_left)
        cycle_buf, frames_this_cycle = start_cycle()
        last_cycle_time = time.time()

    delta_history = []
    # per frame:
    delta_scalar = float(mean_deltas.mean())
    delta_history.append(delta_scalar)

    # per frame: drop bad frames
    ok_var, var_val = frame_passes_variance(frame)
    if not ok_var:
        continue

    # on cycle end:
    ok_motion, avg_delta = cycle_passes_motion(delta_history)
    if not ok_motion:
        print(f"[WARN] Low motion/entropy this cycle (avg delta {avg_delta:.3f}); discarding.")
    else:
        seed = finish_cycle(cycle_buf)
        if not seed_is_new(seed):
            print("[WARN] Seed repeated; discarding.")
        else:
            print("Seed (hex):", seed.hex())
    delta_history = []

    # begins to show cell stats of the top right cell after 10 seconds
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count % 10 == 0:
        s = extract_cell_strats(frame)
        print("CELL(0,0) means/vars: ", s[0, :])
        print("avg mmean-delta per channel: ", mean_deltas.mean(axis=0))

    if cv2.waitKey(1) == ord('g'): # this button press is iffy, sometimes requires double presses
        show_grid = not show_grid
    elif cv2.waitKey(1) == ord('q'): #press q to exit
        break

# release capture and writers
cam.release()
out.release()
cv2.destroyAllWindows()


# things to add:
# - mix with OS randomness?
# - set up camera controls and stability through openCV cam.set
# - set up a system for dumping pre-hash bytes for 1 min, and run a randomness test on it