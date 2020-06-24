# import time
import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image

# from matplotlib import pyplot as plt

# from skimage.metrics import structural_similarity as ssim


WINNER_CAPTURE_LEFT = 592
WINNER_CAPTURE_TOP = 325
WINNER_CAPTURE_RIGHT = 130
WINNER_CAPTURE_BOTTOM = 28
WINNER_CROP = {
    "top": WINNER_CAPTURE_TOP,
    "left": WINNER_CAPTURE_LEFT,
    "width": WINNER_CAPTURE_RIGHT,
    "height": WINNER_CAPTURE_BOTTOM,
}


SCORE_CAPTURE_LEFT_MY_TEAM = 545
SCORE_CAPTURE_TOP_MY_TEAM = 50
SCORE_CAPTURE_RIGHT_MY_TEAM = 42
SCORE_CAPTURE_BOTTOM_MY_TEAM = 45
SCORE_CROP_MY_TEAM = {
    "top": SCORE_CAPTURE_TOP_MY_TEAM,
    "left": SCORE_CAPTURE_LEFT_MY_TEAM,
    "width": SCORE_CAPTURE_RIGHT_MY_TEAM,
    "height": SCORE_CAPTURE_BOTTOM_MY_TEAM,
}
SCORE_CAPTURE_LEFT_THEIR_TEAM = 730
SCORE_CAPTURE_TOP_THEIR_TEAM = 50
SCORE_CAPTURE_RIGHT_THEIR_TEAM = 42
SCORE_CAPTURE_BOTTOM_THEIR_TEAM = 45
SCORE_CROP_THEIR_TEAM = {
    "top": SCORE_CAPTURE_TOP_THEIR_TEAM,
    "left": SCORE_CAPTURE_LEFT_THEIR_TEAM,
    "width": SCORE_CAPTURE_RIGHT_THEIR_TEAM,
    "height": SCORE_CAPTURE_BOTTOM_THEIR_TEAM,
}


def get_my_team(score_screenshot_crop):
    sct = mss.mss()
    screen = np.asarray(sct.grab(score_screenshot_crop))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    return screen.mean()


def display_winner(WINNER_CROP, my_team):
    # Image Gathering
    winning_pixel_count_threshold = 550
    team_that_won = None
    sct = mss.mss()
    screenshot = np.asarray(sct.grab(WINNER_CROP))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    cv2.imshow("Original", screenshot)

    # Window Resizing
    cv2.namedWindow("HSV View Blue", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV View Orange", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Leftover Color Blue", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Leftover Color Orange", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV View Blue", 300, 100)
    cv2.resizeWindow("HSV View Orange", 300, 100)
    cv2.resizeWindow("Leftover Color Blue", 300, 100)
    cv2.resizeWindow("Leftover Color Orange", 300, 100)

    # -- Image Processing -- #
    screen = np.asarray(sct.grab(WINNER_CROP))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    color_filter_lower_bound_blue = np.array([105, 125, 220])
    color_filter_upper_bound_blue = np.array([120, 150, 255])
    color_filter_lower_bound_orange = np.array([3, 123, 230])
    color_filter_upper_bound_orange = np.array([14, 147, 255])

    # All pixels within bounds are white, all pixels out of bounds are black
    screen_mask_blue = cv2.inRange(
        screen, color_filter_lower_bound_blue, color_filter_upper_bound_blue
    )
    screen_mask_orange = cv2.inRange(
        screen, color_filter_lower_bound_orange, color_filter_upper_bound_orange
    )
    cv2.imshow("HSV View Blue", screen_mask_blue)
    cv2.imshow("HSV View Orange", screen_mask_orange)

    # Every pixel that is white in screen_mask_<x> and not black in screen is shown, all others are black
    screen_blue = cv2.bitwise_and(screen, screen, mask=screen_mask_blue)
    screen_orange = cv2.bitwise_and(screen, screen, mask=screen_mask_orange)
    screen_blue_bgr = cv2.cvtColor(screen_blue, cv2.COLOR_HSV2BGR)
    screen_orange_bgr = cv2.cvtColor(screen_orange, cv2.COLOR_HSV2BGR)
    hist_0_channel_blue = np.argwhere(screen_blue_bgr[:, :, 0] > 0)
    hist_0_channel_orange = np.argwhere(screen_orange_bgr[:, :, 0] > 0)
    print(len(hist_0_channel_blue))
    print(len(hist_0_channel_orange))
    if len(hist_0_channel_blue) > winning_pixel_count_threshold:
        team_that_won = "BLUE"
    elif len(hist_0_channel_orange) > winning_pixel_count_threshold:
        team_that_won = "ORANGE"

    # screenshot = screen_orange
    # image_contains_correct_color = 255 in cv2.inRange(screen, color_filter_lower_bound, color_filter_upper_bound)
    # print(image_contains_correct_color)

    screen_blue = cv2.cvtColor(screen_blue, cv2.COLOR_BGR2GRAY)
    screen_orange = cv2.cvtColor(screen_orange, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Leftover Color Blue", screen_blue)
    cv2.imshow("Leftover Color Orange", screen_orange)
    screen = cv2.bitwise_or(screen_blue, screen_orange)
    _, screen = cv2.threshold(screen, 0, 255, cv2.THRESH_BINARY)
    screen = cv2.morphologyEx(screen, cv2.MORPH_OPEN, (5, 5))
    cv2.imshow("Captured View", screen)
    screen = Image.fromarray(screen)
    winner_text = pytesseract.image_to_string(screen)
    try:
        if winner_text == "WINNER" and team_that_won == my_team:
            print(winner_text)
            print("YOU WON!")
        elif winner_text == "WINNER" and team_that_won != my_team:
            print(winner_text)
            print("THEY WON!")
        else:
            winner_text = None
    except UnicodeEncodeError:
        print("That unicode error popped up again")

    return screenshot


def screen_viewfinder(WINNER_CROP):
    # list_of_means = np.array([])
    number = get_my_team(SCORE_CROP_MY_TEAM)

    while True:
        # BLUE: RANGE: 124.24761904761905 to 135.42857142857142
        # ORANGE: RANGE: 153.45820105820107 to 164.1952380952381
        if 90 < number < 140:
            my_team = "BLUE"
        elif 145 < number < 200:
            my_team = "ORANGE"
        else:
            print(f"NUMBER OUT OF RANGE: {number}")

        display_winner(WINNER_CROP, my_team)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

            break
        # elif cv2.waitKey(117) & 0xFF == ord("p"):
        #     cv2.imwrite("go.jpg", scrot)
        #     print("IMAGE SAVED")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    screen_viewfinder(WINNER_CROP)
