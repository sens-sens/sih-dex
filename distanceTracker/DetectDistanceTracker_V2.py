import time
import cv2  # Computer vision library
import numpy as np  # Scientific computing library

# distance from camera to object(face) measured
# centimeter
Known_distance = 100

# width of face in the real world or Object Plane
# centimeter
Known_width = 40


# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame

    # return the distance
    return distance


# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX
# Make sure the video file is in the same directory as your code
# filename = 'edmonton_canada.mp4'
file_size = (640, 480)  # Assumes 1920x1080 mp4 640 x 480

# We want to save the output to a video file
# output_filename = 'edmonton_canada_obj_detect_mobssd.mp4'
# output_frames_per_second = 20.0

RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255

# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# List of categories and classes
categories = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
              4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
              13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'pottedplant', 17: 'sheep', 18: 'sofa',
              19: 'train', 20: 'tvmonitor'}

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))


def refWidth():
    print("**********Inside the refWidth method***************")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1)

    # Process the video
    while cap.isOpened():

        # Capture one frame at a time
        success, frame = cap.read()
        cv2.imshow('width', frame)

        # Do we have a video frame? If true, proceed.
        if success:

            # Capture the frame's height and width
            (h, w) = frame.shape[:2]

            # Create a blob. A blob is a group of connected pixels in a binary
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
            frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), IMG_NORM_RATIO,
                                               RESIZED_DIMENSIONS, 127.5)

            # Set the input for the neural network
            neural_network.setInput(frame_blob)

            # Predict the objects in the image
            neural_network_output = neural_network.forward()

            # Put the bounding boxes around the detected objects
            for i in np.arange(0, neural_network_output.shape[2]):
                # print("value of i: ", i)
                confidence = neural_network_output[0, 0, i, 2]

                # Confidence must be at least 98%
                if confidence > 0.98:
                    idx = int(neural_network_output[0, 0, i, 1])
                    # print("height: width: ", np.array([w,h,w,h]))
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([w, h, w, h])

                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    # print("StartX: ", startX, " start: ", startY, " endX: ", endX, " endY: ", endY)
                    # cm = ((startX + endX) * 2.54) / 96
                    width1 = endX - startX
                    print("\ncm of ref image: ", width1)

            if cv2.waitKey(2000):
                break
    cap.release()
    cv2.destroyAllWindows()
    return width1


def main():
    ref_image_width = refWidth()
    # print(ref_image_width)

    Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_width)
    print("\nfocal length", Focal_length_found)
    # Load a video

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # time.sleep(2)

    for i in range(0, 10):
        # Capture one frame at a time
        success, frame = cap.read()

        # Do we have a video frame? If true, proceed.
    if success:

        # Capture the frame's height and width
        (h, w) = frame.shape[:2]

        # Create a blob. A blob is a group of connected pixels in a binary
        # frame that share some common property (e.g. grayscale value)
        # Preprocess the frame to prepare it for deep learning classification
        frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), IMG_NORM_RATIO,
                                           RESIZED_DIMENSIONS, 127.5)

        # Set the input for the neural network
        neural_network.setInput(frame_blob)

        # Predict the objects in the image
        neural_network_output = neural_network.forward()

        # Put the bounding boxes around the detected objects
        for i in np.arange(0, neural_network_output.shape[2]):
            # print("value of i: ", i)
            confidence = neural_network_output[0, 0, i, 2]

            # Confidence must be at least 98%
            if confidence > 0.98:
                idx = int(neural_network_output[0, 0, i, 1])
                # print("height: width: ", np.array([w,h,w,h]))
                bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                    [w, h, w, h])

                (startX, startY, endX, endY) = bounding_box.astype("int")
            break

            # distance detection

            # width_in_frame = endX - startX

    while cap.isOpened():
        # tracking the person
        tracker_type = 'KCF'
        tracker = cv2.TrackerKCF_create()
        ok, frame = cap.read()
        bbox = (startX, startY, endX, endY)
        ok = tracker.init(frame, bbox)

        while True:
            ok, frame = cap.read()
            # Start timer

            timer = cv2.getTickCount()

            # Update tracker

            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # Draw bounding box

            if ok:

                # Tracking success
                bbox0 = int(bbox[0]) + 100
                bbox2 = int(bbox[2]) - 100
                p1 = (bbox0, int(bbox[1]))
                p2 = (bbox0 + bbox2, int(bbox[1] + bbox[3]))

                # p1 = (int(bbox[0]), int(bbox[1]))
                # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            else:

                # Tracking failure

                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
            print("\nbbox0:", bbox[0])
            print("\nbbox2:", bbox[2])

            newStartX = bbox[0]
            newEndX = bbox[2]
            width_in_frame = newEndX - newStartX
            if width_in_frame != 0:
                Distance = Distance_finder(Focal_length_found, Known_width, width_in_frame)
                print("Distance: ", Distance)
                roundDistance = round(Distance, 2)
                if roundDistance <= 110:
                    engineStatus = "OFF"
                elif roundDistance > 110:
                    engineStatus = "START"
                if newStartX < 50:
                    turn = "Left"
                elif newEndX > 550:
                    turn = "Right"
                roundDistance = str(roundDistance)
                cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (50, 170, 50), 2)
                cv2.putText(frame, "Distance : " + roundDistance + "cm", (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (50, 170, 50), 2)

                cv2.putText(frame, "Engine Status : " + engineStatus, (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (50, 170, 50), 2)
                # cv2.putText(frame, "Turn: " + turn, (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                #             (50, 170, 50), 2)
                frame = cv2.resize(frame, file_size, interpolation=cv2.INTER_NEAREST)
                cv2.imshow('detecting', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


main()
