import cv2

# using default camera 
cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

#run a loop until input to close
while True:
    ret, frame = cam.read()
    out.write(frame)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release capture and writers
cam.release()
out.release()
cv2.destroyAllWindows()