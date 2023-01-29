import cv2
import time
import numpy as np

#This code saves the output in a file: ouput.avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputFile = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#This code starts the webcam. 
cap = cv2.VideoCapture(0)

#This code allows the webcam to start by making the code sleep for 2 seconds.
time.sleep(2)
bg = 0

#This code captures the background for 60 frames. 
for i in range(60):
    ret, bg = cap.read()

#This code flips the background. This code get rid of the mirror-image effect. 
bg = np.flip(bg, axis=1)

#This code reads the captured frames until the camera is open. 
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #This code flips the image for consistency
    img = np.flip(img, axis=1)

    #This code converts the color from BGR to HSV for better detection. 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #This code generates the masks to detect the color red. 
    #THese values can also be arranged by color. 
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    #This code opens and expands the images where there is mask1 (color). 
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    #This code helped us create an inverted mask to segment out the red from the frame. 
    #We are selecting only the part that does not have mask1 and saving it in mask2. 
    mask2 = cv2.bitwise_not(mask1)

    #This code segments the red part out of the frame using bitwise and the inverted mask. 
    #This is keeping only the part of the images without the red color. 
    # **This process will work for any color. 
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    #This code creates an image showing statis background frame pixels only for the masked region. 
    res2 = cv2.bitwise_and(bg, bg, mask=mask1)

    #Finally, we coded to get the video as our output and close all windows. 
    #Here, we are generating the final output by merging res1 and res2. 
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    outputFile.write(finalOutput)
    #Here, we are displaying the output the to the user. 
    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)

cap.release()
outputFile.release()
cv2.destroyAllWindows()

"""
So our algorithm or the steps will be as follows:- 
1. Capture and store the background frame. [ This will be done for some seconds ] 
2. Detect the red colored cloth using color detection and segmentation algorithm. 
3. Segment out the red colored cloth by generating a mask. [ used in code ] 
4. Generate the final augmented output to create a magical effect. [ video.mp4 ]

As we capture frames we are also capturing the colors in those frames. And we need to 
convert the images from BGR (Blue Green Red) to HSV (Hue, Saturation, Value). We need 
to do this so that we can detect the red color more efficiently.

1. Hue: This channel encodes color information. Hue can be thought of as an angle 
where 0 degree corresponds to the red color, 120 degrees corresponds to the green 
color, and 240 degrees corresponds to the blue Student observes and asks questions. 

2. Saturation: This channel encodes the intensity/purity of color. For example, 
pink is less saturated than red. 

3. Value: This channel encodes the brightness of color. Shading and gloss components 
of an image appear in this channel reading the videocapture video.
"""