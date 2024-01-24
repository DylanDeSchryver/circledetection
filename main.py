
import cv2
import numpy as np
import os
directory = os.getcwd()

# Initialize a dictionary to store information about detected circles
detected_circles = {}
path = directory + '/picture.jpg'
def generate_frames():
    frame = cv2.imread(path)

    # Define the coordinates of the rectangle
    r_x, r_y, r_width, r_height = 1200, 1600, 750, 680

    # Draw a rectangle around the rectangle
    cv2.rectangle(frame, (r_x, r_y), (r_x + r_width, r_y + r_height), (255, 0, 0), 2)

    # Crop the frame to the defined rectangle
    r_frame = frame[r_y:r_y+r_height, r_x:r_x+r_width]

    # Convert the cropped frame to grayscale
    gray_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred_frame,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=80,
        param2=60,
        minRadius=100,
        maxRadius=150
    )

    if circles is not None:
        # Convert circle coordinates to global frame
        circles = np.uint16(np.around(circles))
        circles[0, :, 0] += r_x
        circles[0, :, 1] += r_y

        # Update detected_circles dictionary with new circle information
        for i in circles[0, :]:
            circle_id = tuple(i[:2])
            detected_circles[circle_id] = {
                'center': (i[0], i[1]),
                'radius': i[2],
                'frame_counter': 0  # Initialize frame counter
            }

    # Update frame counter for each detected circle
    for circle_id, circle_info in list(detected_circles.items()):
        circle_info['frame_counter'] += 1

        # Draw the outer circle on the original frame
        cv2.circle(frame, circle_info['center'], circle_info['radius'], (0, 255, 0), 2)
        # Draw the center of the circle on the original frame
        cv2.circle(frame, circle_info['center'], 2, (0, 0, 255), 3)


    # Encode the frame

    cv2.imshow('pic', frame)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

generate_frames()
