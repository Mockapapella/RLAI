import os
import time
from pathlib import Path

import cv2
import keyboard
import mss
import numpy as np
from getkeys import get_inputs

from rl_utils.utils import possible_combinations

SCREEN_CAPTURE_LEFT = 0
SCREEN_CAPTURE_TOP = 40
SCREEN_CAPTURE_RIGHT = 1280
SCREEN_CAPTURE_BOTTOM = 740
SCREEN_CROP = {
    "top": SCREEN_CAPTURE_TOP,
    "left": SCREEN_CAPTURE_LEFT,
    "width": SCREEN_CAPTURE_RIGHT,
    "height": SCREEN_CAPTURE_BOTTOM,
}
RESULTS_DICT = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])
TRAINING_WIDTH = 640
TRAINING_HEIGHT = 360
SAMPLE_SIZE = 10000
COUNTDOWN_UNTIL_TRAINING_BEGINS = 5
STARTING_VALUE = 0


def keys_to_output(keys):
    dict_key = ""
    for button_pressed in keys:
        dict_key += button_pressed
    # print(dict_key)

    if dict_key in RESULTS_DICT:
        return RESULTS_DICT[dict_key]
    else:
        return RESULTS_DICT[""]


def Main(file_name, STARTING_VALUE):
    sct = mss.mss()
    training_data = []
    do_nothing_timeout = []
    i = 0

    for i in list(range(COUNTDOWN_UNTIL_TRAINING_BEGINS))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()

    while True:
        screen = np.asarray(sct.grab(SCREEN_CROP))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, (TRAINING_WIDTH, TRAINING_HEIGHT))

        keys = get_inputs()
        output = keys_to_output(keys)

        if output == RESULTS_DICT[""]:
            i += 1
            if len(do_nothing_timeout) <= 100:
                i
                do_nothing_timeout.append(output)
                training_data.append([screen, output])
            print("Length of 'Do Nothing' is {}".format(i))
        else:
            i = 0
            do_nothing_timeout = []
            training_data.append([screen, output])
            print("{} seconds".format(time.time() - last_time))

        last_time = time.time()

        if len(training_data) % (SAMPLE_SIZE / 5) == 0:
            print(len(training_data))
            if len(training_data) == SAMPLE_SIZE:
                np.save(file_name, training_data)
                print("SAVED")
                training_data = []
                STARTING_VALUE += 1
                file_name = Path(
                    "training_data/rgb/640x360-{}_Sample_Size/training_data-{}.npy".format(
                        SAMPLE_SIZE, STARTING_VALUE
                    )
                )

        if keyboard.is_pressed("P"):
            for i in list(range(30))[::-1]:
                print(i + 1)
                time.sleep(1)


if __name__ == "__main__":

    file_name = Path(
        "training_data/rgb/640x360-{}_Sample_Size/training_data-{}.npy".format(
            SAMPLE_SIZE, STARTING_VALUE
        )
    )

    if os.path.isfile(file_name):
        print("File exists, loading previous data!")
        with list(np.load(file_name)) as data:
            training_data = data
    else:
        print("File does not exist, starting fresh!")
        training_data = []

    Main(file_name, STARTING_VALUE)
