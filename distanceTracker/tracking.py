import cv2
import sys
import time


tracker_type ='KCF'
tracker = cv2.TrackerKCF_create()
              # Read video
video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(2)
ok, frame = video.read()
bbox = (200, 100, 250, 360)
ok = tracker.init(frame, bbox)
while True:
    ok, frame = video.read()
    frame = cv2.flip(frame, 1)
    # Start timer

    timer = cv2.getTickCount()

    # Update tracker

    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box

    if ok:

    # Tracking success

        p1 = (int(bbox[0]), int(bbox[1]))

        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    else:

        # Tracking failure

        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame

    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result

    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed

    k = cv2.waitKey(1) & 0xff

    if k == 27:
        break
#(x,y,w,h)
