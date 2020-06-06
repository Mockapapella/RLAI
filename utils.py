from itertools import chain, combinations


def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def possible_combinations(choices):
    results = []
    results_binary = []
    results_dict = {}
    iterator = 0

    powerset_results = list(powerset(choices))

    for powerset_results in list(powerset(choices)):
        button_choice_list = []
        for result in powerset_results:
            button_choice_list.append(result)
        results.append(button_choice_list)

    for i in range(len(results)):
        button_choice_list = []
        for j in range(len(results)):
            if i == j:
                button_choice_list.append(1)
            else:
                button_choice_list.append(0)
        results_binary.append(button_choice_list)

    for button_set in results:
        dict_key = ""
        for string in button_set:
            dict_key += string
        # print(button_set)
        results_dict[dict_key] = results_binary[iterator]
        iterator += 1

    # print(results_binary)
    # print(results)
    # print(results_dict)
    return results_dict


def string_to_commands(choices):
    results = []
    results_binary = []
    results_dict = {}
    iterator = 0

    powerset_results = list(powerset(choices))

    for powerset_results in list(powerset(choices)):
        button_choice_list = []
        for result in powerset_results:
            button_choice_list.append(result)
        results.append(button_choice_list)

    for button_set in results:
        dict_key = ""
        for string in button_set:
            dict_key += string
        results_dict[dict_key] = results[iterator]
        iterator += 1

    # for result in results_dict:
    #     print("{}:{}".format(result, results_dict[result]))

    # print(results_dict)
    return results_dict

# Done by Frannecklp

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype="uint8")
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
