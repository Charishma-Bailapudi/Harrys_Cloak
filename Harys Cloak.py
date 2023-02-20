import cv2
import time
import numpy as np
cap = cv2.VideoCapture(0)

ret, background = cap.read()
time.sleep(2)
ret, background = cap.read()


#define all the kernels size
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Function for remove noise from mask
def filter_mask(mask):

    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)

    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)

    return dilation


while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([100,33,101])
    upper_bound = np.array([141,139,243])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    cv2.imshow("mask",mask)
    # Filter mask
    mask = filter_mask(mask)
    # Apply the mask to take only those region from the saved background where our cloak is present in the current frame
    cloak = cv2.bitwise_and(background, background, mask=mask)

    # create inverse mask
    inverse_mask = cv2.bitwise_not(mask)

    # Apply the inverse mask to take those region of the current frame where cloak is # not present
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    combined = cv2.add(cloak, current_background)

    cv2.imshow("Final",combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()