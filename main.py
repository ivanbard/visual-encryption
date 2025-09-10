import cv2

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

def prep_frame(frame):
    return cv2.resizeWindow(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

def cell_dimensions():
    CELL_W = PROC_W // GRID_W
    CELL_H = PROC_H // GRID_H
    return CELL_W, CELL_H

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

#run a loop until input to close
while True:
    ret, frame = cam.read()
    out.write(frame)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'): #press q to exit
        break

# release capture and writers
cam.release()
out.release()
cv2.destroyAllWindows()