import numpy as np
import cv2
import time
import os
import mouse
import keyboard

from pathlib import Path

from grabscreen import grab_screen
from getkeys import get_inputs
from possible_combinations import possible_combinations

x_1 = 0
y_1 = 40
x_2 = 1280
y_2 = 740
results_dict = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])
WIDTH = 640
HEIGHT = 360
SAMPLE_SIZE = 10000

starting_value = 0


def keys_to_output(keys):
    dict_key = ""
    for button_pressed in keys:
        dict_key += button_pressed
    # print(dict_key)

    if dict_key in results_dict:
        return results_dict[dict_key]
    else:
        return results_dict[""]


file_name = Path(
    "training_data/rgb/640x360-{}_Sample_Size/training_data-{}.npy".format(
        SAMPLE_SIZE, starting_value
    )
)

if os.path.isfile(file_name):
    print("File exists, loading previous data!")
    with list(np.load(file_name)) as data:
        training_data = data
else:
    print("File does not exist, starting fresh!")
    training_data = []


def Main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    do_nothing_timeout = []
    i = 0

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()

    while True:
        screen = grab_screen(region=(x_1, y_1, x_2, y_2))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, (WIDTH, HEIGHT))

        keys = get_inputs()
        output = keys_to_output(keys)

        if output == results_dict[""]:
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
                starting_value += 1
                file_name = Path(
                    "training_data/rgb/640x360-{}_Sample_Size/training_data-{}.npy".format(
                        SAMPLE_SIZE, starting_value
                    )
                )

        if keyboard.is_pressed("P"):
            for i in list(range(30))[::-1]:
                print(i + 1)
                time.sleep(1)


if __name__ == "__main__":
    Main(file_name, starting_value)
