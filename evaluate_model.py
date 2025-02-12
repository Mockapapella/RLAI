import cv2
import mss
import numpy as np
import torch

from rl_utils.models import MnistNet

# x_1 = 0
# y_1 = 40
# x_2 = 1280
# y_2 = 740
# results_dict = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])
# dict_keys_to_button_presses_dict = string_to_commands(
#     ["shift", "a", "d", "s", "w", "left", "right"]
# )

# WIDTH = 640
# HEIGHT = 360
# LR = 1e-3
# EPOCHS = 50
# MODEL_NAME = "Rocket-Python-{}-{}x{}-{}-{}_Epochs.model".format(
#     "otherception3", WIDTH, HEIGHT, LR, EPOCHS
# )


CUDA_GPU_TO_USE = 1
SCORE_DETECTION_PATH = "./ai/models/score_detector/model.pth"

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

# -- Initialize Models -- #
CUDA_GPU_TO_USE = 1
DEVICE = torch.device("cuda:{}".format(CUDA_GPU_TO_USE) if torch.cuda.is_available() else "cpu")

# Score Detection
SCORE_DETECTOR = MnistNet()
SCORE_DETECTOR.load_state_dict(torch.load(SCORE_DETECTION_PATH))
SCORE_DETECTOR.to(DEVICE)

GAME_SCORE = [0, 0]


def score(score_screenshot_crop, DEVICE, mnist=False):
    sct = mss.mss()
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


# def keys_to_press(move):
#     dict_key = ""
#     for value in results_dict:
#         if move == results_dict[value]:
#             # print("##############################")
#             dict_key = value
#     print(dict_key)

#     for button in dict_keys_to_button_presses_dict["shiftadswleftright"]:
#         try:
#             mouse.release(button)
#         except:
#             pass
#         try:
#             keyboard.release(button)
#         except:
#             pass

#     for button_press in dict_keys_to_button_presses_dict:
#         if dict_key == button_press:
#             for button in dict_keys_to_button_presses_dict[button_press]:
#                 try:
#                     mouse.press(button)
#                 except:
#                     pass
#                 try:
#                     keyboard.press(button)
#                 except:
#                     pass


# def unpress_all_keys():
#     for button in dict_keys_to_button_presses_dict["shiftadswleftright"]:
#         try:
#             mouse.release(button)
#         except:
#             pass
#         try:
#             keyboard.release(button)
#         except:
#             pass


def Main(SCORE_CROP_MY_TEAM, SCORE_CROP_THEIR_TEAM, DEVICE, GAME_SCORE, mnist=True):
    """
    This function is broken until further notice
    """
    # new_game = True
    list_of_scores_since_last_goal = np.array([GAME_SCORE])
    while True:
        my_score = score(SCORE_CROP_MY_TEAM, DEVICE, mnist=True)
        their_score = score(SCORE_CROP_THEIR_TEAM, DEVICE, mnist=True)
        current_score_count = np.array([my_score[0].item(), their_score[0].item()])
        list_of_scores_since_last_goal = np.vstack(
            ([list_of_scores_since_last_goal, current_score_count])
        )
        # if new_game:
        #     print("NEW GAME")
        #     print(GAME_SCORE)
        #     new_game = False
        # if all(elem in GAME_SCORE for elem in current_score_count):
        #     print(current_score_count)
        #     GAME_SCORE = current_score_count
        # if game ends reset the score

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    # Everything below this point is part of the old AI, but still has useful parts moving forward. Will refactor later.
    # for i in list(range(4))[::-1]:
    #     print(i + 1)
    #     time.sleep(1)

    # last_time = time.time()

    # paused = False
    # while True:
    #     if not paused:
    #         screen = grab_screen(region=(x_1, y_1, x_2, y_2))
    #         screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    #         screen = cv2.resize(screen, (WIDTH, HEIGHT))

    #         # print("{} seconds".format(time.time()-last_time))
    #         last_time = time.time()

    #         with tf.Session(
    #             config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    #         ):
    #             prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0]
    #         moves = list(np.around(prediction))
    #         # print(moves, prediction)
    #         keys_to_press(moves)

    #     keys = get_inputs()

    #     if keyboard.is_pressed("P"):
    #         if paused:
    #             print("UNPAUSED")
    #             paused = False
    #             time.sleep(1)
    #         else:
    #             print("PAUSED")
    #             paused = True
    #             unpress_all_keys()
    #             time.sleep(1)


if __name__ == "__main__":
    Main(SCORE_CROP_MY_TEAM, SCORE_CROP_THEIR_TEAM, DEVICE, GAME_SCORE)
