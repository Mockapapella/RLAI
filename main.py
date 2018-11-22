import numpy as np
import cv2
import time
import os
import mouse

from grabscreen import grab_screen
from getkeys import key_mouse_check
from possible_combinations import possible_combinations

x_1 = 0
y_1 = 40
x_2 = 1280
y_2 = 740
results_dict = possible_combinations(['shift','a','d','s','w',' ','left','right'])

def keys_to_output(keys):
    dict_key = ""
    for button_pressed in keys:
        dict_key += button_pressed

    if dict_key in results_dict:
        return results_dict[dict_key]
    else:
        return results_dict[""]

file_name = "training_data.npy"

if os.path.isfile(file_name):
    print("File exists, loading previous data!")
    training_data = list(np.load(file_name))
else:
    print("File does not exist, starting fresh!")
    training_data = []
total_combos = []

def Main():
    for i in list(range(4))[::-1]:
      print(i+1)
      time.sleep(1)

    last_time = time.time()

    while(True):
        screen = grab_screen(region=(x_1,y_1,x_2,y_2))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160, 90))
        keys = key_mouse_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        print("{} seconds".format(time.time()-last_time))
        # last_time = time.time()
        print(output)

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

if __name__ == "__main__":
    Main()