#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from pathlib import Path

import threading
from queue import Queue


from possible_combinations import possible_combinations

permutation_dict = possible_combinations(["shift", "a", "d", "s", "w", "left", "right"])

ai_width = 640
ai_height = 360
starting_value = 1
ending_value = 430
file_error_value = 0
total_samples = 0

file_name = Path(
    "training_data/rgb/640_360/training_data-{}-{}-{}.npy".format(
        ai_width, ai_height, starting_value
    )
)

while starting_value < ending_value + 1:
    try:
        train_data = np.load(file_name)
        print(len(train_data))
        # print(type(train_data))

        df = pd.DataFrame(train_data)
        print(df.head())
        # print(Counter(df[1].apply(str)))

        final_data = []
        final_data_prime = []
        replacement_data = []
        unique_presses = []

        for data in train_data:
            replacement_data.append(data[1])

        for data in replacement_data:
            if data not in unique_presses:
                unique_presses.append(data)

        i = 0
        j = 0
        for data in train_data:
            for press in unique_presses:
                if data[1] == press:
                    if final_data_prime.count(press) <= 40:
                        final_data.append(data)
                        final_data_prime.append(press)
                        i += 1
            j += 1
            print("{}:{}".format(j, i))

        print(len(final_data))
        total_samples += len(final_data)

        # df = pd.DataFrame(final_data)
        # print(df.head())

        shuffle(final_data)
        print(
            "Saving {} of {}".format(
                starting_value - file_error_value, ending_value - file_error_value
            )
        )
        np.save(
            Path(
                "training_data/rgb/640_360_balanced/training_data-{}-{}-{}.npy".format(
                    ai_width, ai_height, starting_value - file_error_value
                )
            ),
            final_data,
        )

        starting_value += 1
        file_name = Path(
            "training_data/rgb/640_360/training_data-{}-{}-{}.npy".format(
                ai_width, ai_height, starting_value
            )
        )
    except:
        starting_value += 1
        file_name = Path(
            "training_data/rgb/640_360/training_data-{}-{}-{}.npy".format(
                ai_width, ai_height, starting_value
            )
        )
        file_error_value += 1

print("Total samples: {}".format(total_samples))
print("Average samples: {}".format(total_samples / (starting_value - file_error_value)))
