import cv2
import numpy as np

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

show_grid = True

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

    # begins to show cell stats of the top right cell after 10 seconds
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count % 10 == 0:
        s = extract_cell_strats(frame)
        print("CELL(0,0) means/vars: ", s[0, :])

    if cv2.waitKey(1) == ord('g'): # this button press is iffy, sometimes requires double presses
        show_grid = not show_grid
    elif cv2.waitKey(1) == ord('q'): #press q to exit
        break

# release capture and writers
cam.release()
out.release()
cv2.destroyAllWindows()