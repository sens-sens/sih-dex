# Project: How to Detect Objects in Video Using MobileNet SSD in OpenCV
# Author: Addison Sears-Collins
# Date created: March 1, 2021
# Description: Object detection using OpenCV
import time
import cv2  # Computer vision library
import numpy as np  # Scientific computing library

# distance from camera to object(body) measured
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
file_size = (640, 480)  # Assumes 1920x1080 mp4

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
    time.sleep(2)

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
                    #cm = ((startX + endX) * 2.54) / 96
                    width = endX - startX
                    print("\ncm of ref image: ", width)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    return width

    cap.release()
    cv2.destroyAllWindows()


def main():
    ref_image_width = refWidth()
    print(ref_image_width)
    Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_width)
    print("\nfocal length", Focal_length_found)
    # Load a video

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2)

    # Process the video
    while cap.isOpened():

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

                    #distance detection

                    width_in_frame =  endX - startX

                    print(width_in_frame)
                    if width_in_frame != 0:
                        Distance = Distance_finder(Focal_length_found, Known_width, width_in_frame)
                        print("Distance: ", Distance)
                        roundDistance = round(Distance, 2)
                    label = "{}: {:.2f}% -- Distance: {:.2f}cm".format(classes[idx], confidence * 100, Distance)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), bbox_colors[idx], 2)
                    #cm = ((startX + endX) * 2.54) / 96

                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colors[idx], 2)

            # We now need to resize the frame so its dimensions
            # are equivalent to the dimensions of the original frame
            frame = cv2.resize(frame, file_size, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('detecting', frame)
            # Write the frame to the output video file

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # No more video frames left
        else:
            break

    # # Release the video recording
    cap.release()
    cv2.destroyAllWindows()



main()
