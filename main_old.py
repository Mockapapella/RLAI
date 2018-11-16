import numpy as np
import cv2
import time
import pyautogui

from grabscreen import grab_screen
from getkeys import key_check

W="w"
A="a"
S="s"
D="d"
left_side=500-400
right_side=780+400
top_side=270
bottom_side=550
x_1 = 0
y_1 = 40
x_2 = 1280
y_2 = 740

averaging_array = []

def keys_to_output(keys):
    #[A,W,D]
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[0] = 1

    return output





def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_circles(img, circles):
    global direction
    try:
        for circle in circles:
            center = circle[0]
            cv2.circle(img, (center[0], center[1]), center[2], (0, 255, 0), 4)
            cv2.line(img, (center[0], center[1]), (center[0], center[1]), (255,0,0), 4)
            if center[0] < (x_2/2)-50:
                return "left"
            elif center[0] > (x_2/2)+50:
                return "right"
            else:
                return "center"
    except:
        pass

def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200, threshold2=350)
    processed_image = cv2.GaussianBlur(processed_image, (7,7), 0)
    vertices = np.array([[left_side,top_side],[right_side,top_side],[right_side,bottom_side],[left_side,bottom_side]])
    processed_image = roi(processed_image, [vertices])

    #cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,,1000,minRadius=0,maxRadius=75)
    circles = cv2.HoughCircles(processed_image,cv2.HOUGH_GRADIENT,2,25,minRadius=0,maxRadius=150)
    direction = draw_circles(original_image,circles)
    cv2.line(original_image, (int(x_2/2), 0), (int(x_2/2), y_2), (0,0,0), 3)

    return original_image, direction

def PressKey(button):
    pyautogui.keyDown(button)

def ReleaseKey(button):
    pyautogui.keyUp(button)

def Center():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def Left():
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)

def Right():
    PressKey(D)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)

def slow_down():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

for i in list(range(4))[::-1]:
  print(i+1)
  time.sleep(1)

last_time = time.time()
while(True):
    screen = grab_screen(region=(x_1,y_1,x_2,y_2))
    new_screen, direction = process_image(screen)
    print("Loop took {} seconds".format(time.time()-last_time))
    averaging_array.append(time.time()-last_time)
    last_time = time.time()
    # cv2.imshow("Window", new_screen)
    cv2.imshow("Rocket Python", cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))

    if direction == "left":
        Left()
    elif direction == "right":
        Right()
    elif direction == "center":
        Center()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        print(sum(averaging_array)/len(averaging_array))
        break