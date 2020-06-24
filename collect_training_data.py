"""
This file collects training data and saves it to a numpy array.

TODO:
- Add full controller input
- Add full screen capture
"""
import os
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from rl_utils.models import MnistNet
from rl_utils.utils import possible_combinations
from rl_utils.utils import ThreadingWrite

# import threading
# from pprint import pprint
# import keyboard
# from rl_utils.getkeys import get_inputs

# AI Config
CUDA_GPU_TO_USE = 1
SCORE_DETECTION_PATH = "./ai/models/score_detector/model20.pth"

# -- Initialize Models -- #
CUDA_GPU_TO_USE = 1
DEVICE = torch.device("cuda:{}".format(CUDA_GPU_TO_USE) if torch.cuda.is_available() else "cpu")

# Score Detection
SCORE_DETECTOR = MnistNet()
SCORE_DETECTOR.load_state_dict(torch.load(SCORE_DETECTION_PATH))
SCORE_DETECTOR.to(DEVICE)

# Score Capture Zones
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

# Winner Capture Zone
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

# Round Countdown Zone
START_ROUND_CAPTURE_LEFT = 610
START_ROUND_CAPTURE_TOP = 240
START_ROUND_CAPTURE_RIGHT = 100
START_ROUND_CAPTURE_BOTTOM = 100
START_ROUND_CROP = {
    "top": START_ROUND_CAPTURE_TOP,
    "left": START_ROUND_CAPTURE_LEFT,
    "width": START_ROUND_CAPTURE_RIGHT,
    "height": START_ROUND_CAPTURE_BOTTOM,
}

# Training Data Config
RESULTS_DICT = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])
TRAINING_WIDTH = 640
TRAINING_HEIGHT = 360
COUNTDOWN_UNTIL_TRAINING_BEGINS = 0
DATA_TITLE = "Carlaisle"

# Create labeled folder path if it doesn't already exist
TRAINING_DATA_FOLDER_PATH = "ai/data/{}/{}_width-{}_height/".format(
    DATA_TITLE, TRAINING_WIDTH, TRAINING_HEIGHT
)
Path(TRAINING_DATA_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
LIST_OF_TRAINING_SAMPLES = os.listdir(TRAINING_DATA_FOLDER_PATH)


# If the folder path already contains training data samples, make sure the next sample is incremented by 1
if len(LIST_OF_TRAINING_SAMPLES) == 0:
    TRAINING_DATA_SEQUENCE_COUNTER = 0
elif len(LIST_OF_TRAINING_SAMPLES) > 0:
    latest_sample = LIST_OF_TRAINING_SAMPLES[-1]
    latest_sample_number = int(latest_sample.split("-")[-1].split(".")[0])
    TRAINING_DATA_SEQUENCE_COUNTER = latest_sample_number

# exit()


def get_my_team(sct, score_screenshot_crop):
    screen = np.asarray(sct.grab(score_screenshot_crop))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    return screen.mean()


def check_win_loss_status(sct, WINNER_CROP, my_team):
    # Image Gathering
    winning_pixel_count_threshold = 550
    team_that_won = None

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

    # Every pixel that is white in screen_mask_<x> and not black in screen is shown, all others are black
    screen_blue = cv2.bitwise_and(screen, screen, mask=screen_mask_blue)
    screen_orange = cv2.bitwise_and(screen, screen, mask=screen_mask_orange)
    screen_blue_bgr = cv2.cvtColor(screen_blue, cv2.COLOR_HSV2BGR)
    screen_orange_bgr = cv2.cvtColor(screen_orange, cv2.COLOR_HSV2BGR)
    hist_0_channel_blue = np.argwhere(screen_blue_bgr[:, :, 0] > 0)
    hist_0_channel_orange = np.argwhere(screen_orange_bgr[:, :, 0] > 0)
    if len(hist_0_channel_blue) > winning_pixel_count_threshold:
        team_that_won = "BLUE"
    elif len(hist_0_channel_orange) > winning_pixel_count_threshold:
        team_that_won = "ORANGE"

    screen_blue = cv2.cvtColor(screen_blue, cv2.COLOR_BGR2GRAY)
    screen_orange = cv2.cvtColor(screen_orange, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Leftover Color Blue", screen_blue)
    # cv2.imshow("Leftover Color Orange", screen_orange)
    screen = cv2.bitwise_or(screen_blue, screen_orange)
    _, screen = cv2.threshold(screen, 0, 255, cv2.THRESH_BINARY)
    screen = cv2.morphologyEx(screen, cv2.MORPH_OPEN, (5, 5))
    # cv2.imshow("Captured View", screen)
    screen = Image.fromarray(screen)
    winner_text = pytesseract.image_to_string(screen)
    try:
        if winner_text == "WINNER" and team_that_won == my_team:
            print(winner_text)
            print("YOU WON THE GAME!")
            return "WIN"
        elif winner_text == "WINNER" and team_that_won != my_team:
            print(winner_text)
            print("LOSS")
            return team_that_won
        else:
            winner_text = None
            return None
    except UnicodeEncodeError:
        print("That unicode error popped up again")


def keys_to_output(keys):
    dict_key = ""
    for button_pressed in keys:
        dict_key += button_pressed
    # print(dict_key)

    if dict_key in RESULTS_DICT:
        return RESULTS_DICT[dict_key]
    else:
        return RESULTS_DICT[""]


def score(sct, score_screenshot_crop, DEVICE, mnist=False):
    screen = np.asarray(sct.grab(score_screenshot_crop))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _, screen = cv2.threshold(screen, 200, 255, cv2.THRESH_BINARY)
    screen = cv2.dilate(screen, (3, 3), iterations=3)
    screen = cv2.resize(screen, (28, 28))
    # cv2.imshow("Captured View {}".format(str(score_screenshot_crop)), screen)

    if mnist is True:
        score_tensor = torch.tensor(screen, dtype=torch.float).to(DEVICE)
        score_tensor = score_tensor[None, None].to(DEVICE)
        score_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = SCORE_DETECTOR(score_tensor).to(DEVICE)
            _, prediction = torch.max(outputs.data, 1)
            # print(prediction)
            return prediction
    elif mnist is False:
        return None


def round_start_checker(sct, START_ROUND_CROP):
    # Image Gathering
    template = cv2.imread("rl_utils/3.jpg", 0)

    # -- Image Processing -- #
    screen = np.asarray(sct.grab(START_ROUND_CROP))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    color_filter_lower_bound = np.array([20, 100, 220])
    color_filter_upper_bound = np.array([30, 130, 255])

    # All pixels within bounds are white, all pixels out of bounds are black
    screen_mask = cv2.inRange(screen, color_filter_lower_bound, color_filter_upper_bound)

    # Every pixel that is white in screen_mask and not black in screen is shown, all others are black
    screen = cv2.bitwise_and(screen, screen, mask=screen_mask)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _, screen = cv2.threshold(screen, 0, 255, cv2.THRESH_BINARY)
    screen = cv2.morphologyEx(screen, cv2.MORPH_OPEN, (3, 3))
    # cv2.imshow("Captured View", screen)

    # Similarity Matching
    similarity = ssim(screen, template)
    # if similarity > 0.88:
    #     print("3: %.2f" % similarity)

    return similarity


def Main(
    TRAINING_DATA_SEQUENCE_COUNTER,
    DATA_TITLE,
    SCORE_CROP_MY_TEAM,
    SCORE_CROP_THEIR_TEAM,
    START_ROUND_CROP,
    WINNER_CROP,
):
    # Constant Variables
    sct = mss.mss()
    similarity_threshold = 0.88
    score_length = 20
    # start_time = time.time()

    # Variable Variables
    in_game = False
    in_round = False
    start_of_round = False
    game_start = False
    similarity = 0
    estimated_current_score = None
    remembered_score = torch.Tensor([[-1, -1]]).to(DEVICE)
    round_training_data = np.array([])
    # list_of_game_scores = np.array([])
    rolling_list_of_scores = pd.DataFrame(remembered_score)
    training_data_filename = "{}_training_data-{}.npy".format(
        DATA_TITLE, TRAINING_DATA_SEQUENCE_COUNTER
    )

    for i in list(range(COUNTDOWN_UNTIL_TRAINING_BEGINS))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("MONITORING")

    while True:

        # Get the current score
        my_score = score(sct, SCORE_CROP_MY_TEAM, DEVICE, mnist=True)
        their_score = score(sct, SCORE_CROP_THEIR_TEAM, DEVICE, mnist=True)
        score_count = [[my_score, their_score]]
        rolling_list_of_scores = rolling_list_of_scores.append(score_count)
        rolling_list_of_scores = rolling_list_of_scores[-score_length:]
        estimated_current_score = rolling_list_of_scores.mode(axis=0)
        estimated_current_score = torch.Tensor(
            [[estimated_current_score[0][0], estimated_current_score[1][0]]]
        ).to(DEVICE)

        """
        TODO: Check if the game has ended
        """
        if in_game is False:
            print("WAITING FOR NEXT GAME")
            # Detect if RL has placed us in a new or low scoring ongoing game
            # TODO: Check for color profiles (from histograms) for better redundancy
            if (
                0 <= estimated_current_score[0][0].item() <= 5
                and 0 <= estimated_current_score[0][1].item() <= 5
            ):
                in_game = True
                game_start = True
                number = get_my_team(sct, SCORE_CROP_MY_TEAM)
                # BLUE: RANGE: 124.24761904761905 to 135.42857142857142
                # ORANGE: RANGE: 153.45820105820107 to 164.1952380952381
                if 90 < number < 140:
                    my_team = "BLUE"
                elif 140 <= number < 200:
                    my_team = "ORANGE"
                else:
                    print(f"NUMBER OUT OF RANGE: {number}")

        elif in_game is True:
            if game_start:
                print("GAME START")

            # Start of a new round
            if similarity > similarity_threshold and start_of_round is False or game_start is True:
                print("ROUND START")
                in_round = True
                start_of_round = True
                TRAINING_DATA_SEQUENCE_COUNTER += 1
                round_training_data = np.array([])
                training_data_filename = "{}_training_data-{}.npy".format(
                    DATA_TITLE, TRAINING_DATA_SEQUENCE_COUNTER
                )
                remembered_score = estimated_current_score
            elif (
                not torch.eq(estimated_current_score, remembered_score).all()
                and start_of_round is True
            ):
                print("ROUND END")
                in_round = False
                start_of_round = False

            if in_round is True:
                print("IN ROUND")
                round_training_data = np.append(round_training_data, score_count)
            elif in_round is False:
                print("WAITING FOR NEXT ROUND")
                # Append this `if` statement data to an array, only save it once the game is over
                if (
                    not Path(
                        os.path.join(TRAINING_DATA_FOLDER_PATH, training_data_filename)
                    ).is_file()
                    and len(round_training_data) > 0
                ):
                    write_training_data_to_file = ThreadingWrite(
                        round_training_data, TRAINING_DATA_FOLDER_PATH, training_data_filename
                    )
                    write_training_data_to_file.start()
                similarity = round_start_checker(sct, START_ROUND_CROP)
                win_loss_status = check_win_loss_status(sct, WINNER_CROP, my_team)
                if win_loss_status == "WIN":
                    in_game = False
                elif win_loss_status == "LOSS":
                    in_game = False
                if in_game is False:
                    print("DATA SAVED")
                    print("INITIALIZATION VARIABLES RESET")
            game_start = False

        # screen = np.asarray(sct.grab(SCREEN_CROP))
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # screen = cv2.resize(screen, (TRAINING_WIDTH, TRAINING_HEIGHT))

        # keys = get_inputs()
        # output = keys_to_output(keys)

        # if output == RESULTS_DICT[""]:
        #     i += 1
        #     if len(do_nothing_timeout) <= 100:
        #         i
        #         do_nothing_timeout.append(output)
        #         training_data.append([screen, output])
        #     print("Length of 'Do Nothing' is {}".format(i))
        # else:
        #     i = 0
        #     do_nothing_timeout = []
        #     training_data.append([screen, output])
        #     print("{} seconds".format(time.time() - last_time))

        # last_time = time.time()

        # if len(training_data) % (SAMPLE_SIZE / 5) == 0:
        #     print(len(training_data))
        #     if len(training_data) == SAMPLE_SIZE:
        #         np.save(TRAINING_DATA_FILENAME_STARTING_POINT, training_data)
        #         print("SAVED")
        #         training_data = []
        #         TRAINING_DATA_SEQUENCE_COUNTER += 1
        #         TRAINING_DATA_FILENAME_STARTING_POINT = Path(
        #             "training_data/rgb/640x360-{}_Sample_Size/training_data-{}.npy".format(
        #                 SAMPLE_SIZE, TRAINING_DATA_SEQUENCE_COUNTER
        #             )
        #         )

        if cv2.waitKey(25) & 0xFF == ord("q"):
            # print("Mean: ", np.mean(list_of_similarities))
            cv2.destroyAllWindows()
            break

        # if keyboard.is_pressed("P"):
        #     for i in list(range(30))[::-1]:
        #         print(i + 1)
        #         time.sleep(1)


if __name__ == "__main__":
    Main(
        TRAINING_DATA_SEQUENCE_COUNTER,
        DATA_TITLE,
        SCORE_CROP_MY_TEAM,
        SCORE_CROP_THEIR_TEAM,
        START_ROUND_CROP,
        WINNER_CROP,
    )
