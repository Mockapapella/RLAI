import time

import cv2
import keyboard
import mouse
import numpy as np
import tensorflow as tf

from grabscreen import grab_screen
from models import otherception3
from possible_combinations import possible_combinations
from possible_combinations import string_to_commands

# from getkeys import get_inputs

x_1 = 0
y_1 = 40
x_2 = 1280
y_2 = 740
results_dict = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])
dict_keys_to_button_presses_dict = string_to_commands(
    ["shift", "a", "d", "s", "w", "left", "right"]
)

WIDTH = 640
HEIGHT = 360
LR = 1e-3
EPOCHS = 50
MODEL_NAME = "Rocket-Python-{}-{}x{}-{}-{}_Epochs.model".format(
    "otherception3", WIDTH, HEIGHT, LR, EPOCHS
)


def keys_to_press(move):
    dict_key = ""
    for value in results_dict:
        if move == results_dict[value]:
            # print("##############################")
            dict_key = value
    print(dict_key)

    for button in dict_keys_to_button_presses_dict["shiftadswleftright"]:
        try:
            mouse.release(button)
        except Exception:
            pass
        try:
            keyboard.release(button)
        except Exception:
            pass

    for button_press in dict_keys_to_button_presses_dict:
        if dict_key == button_press:
            for button in dict_keys_to_button_presses_dict[button_press]:
                try:
                    mouse.press(button)
                except Exception:
                    pass
                try:
                    keyboard.press(button)
                except Exception:
                    pass


def unpress_all_keys():
    for button in dict_keys_to_button_presses_dict["shiftadswleftright"]:
        try:
            mouse.release(button)
        except Exception:
            pass
        try:
            keyboard.release(button)
        except Exception:
            pass


model = otherception3(WIDTH, HEIGHT, 3, LR, output=128)
model.load(MODEL_NAME)


def Main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    # last_time = time.time()

    paused = False
    while True:
        if not paused:
            screen = grab_screen(region=(x_1, y_1, x_2, y_2))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            # print("{} seconds".format(time.time()-last_time))
            # last_time = time.time()

            with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            ):
                prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0]
            moves = list(np.around(prediction))
            # print(moves, prediction)
            keys_to_press(moves)

        # keys = get_inputs()

        if keyboard.is_pressed("P"):
            if paused:
                print("UNPAUSED")
                paused = False
                time.sleep(1)
            else:
                print("PAUSED")
                paused = True
                unpress_all_keys()
                time.sleep(1)


if __name__ == "__main__":
    Main()
