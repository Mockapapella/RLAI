# train_model.py
from pathlib import Path
from random import shuffle

import numpy as np
import tensorflow as tf

from models import otherception3

FILE_I_END = 185

WIDTH = 640
HEIGHT = 360
LR = 1e-3
EPOCHS = 50

# modelpath = Path("models/640_360_RGB_BALANCED/rocket-python-{}-{}-{}-epochs.model".format(LR, "otherception3", EPOCHS))
MODEL_NAME = "Rocket-Python-{}-{}x{}-{}-{}_Epochs.model".format(
    "otherception3", WIDTH, HEIGHT, LR, EPOCHS
)
PREV_MODEL = "Rocket-Python-{}-{}x{}-{}-{}_Epochs.model".format(
    "otherception3", WIDTH, HEIGHT, LR, EPOCHS
)

LOAD_MODEL = False

with Path("training_data/rgb/640x360-2000_Sample_Size/training_data-1.npy") as get_outputs:
    total_outputs = np.load(get_outputs)
    print(len(total_outputs[1][1]))
    total_outputs = len(total_outputs[1][1])

model = otherception3(WIDTH, HEIGHT, 3, LR, output=total_outputs)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print("We have loaded a previous model!!!!")

for e in range(EPOCHS):
    data_order = [i for i in range(1, FILE_I_END + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            # filepath = Path("training_data/rgb/640_360_balanced/training_data-{}-{}-{}.npy".format(WIDTH, HEIGHT, i))
            # train_data = np.load(str(filepath))
            print(
                """
                ########################################################
                # NOW TRAINING FILE {} AND COUNT {} ON EPOCH NUMBER {} #
                ########################################################
                """.format(
                    i, count, e
                )
            )
            train_data = np.load(
                "training_data/rgb/640x360-2000_Sample_Size/training_data-{}.npy".format(i)
            )
            shuffle(train_data)
            print("training_data-{}.npy".format(i), len(train_data))

            limit_of_test_data = int(round(0.05 * len(train_data)))

            train = train_data[:-limit_of_test_data]
            test = train_data[-limit_of_test_data:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
            test_y = [i[1] for i in test]

            with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            ):
                model.fit(
                    X,
                    Y,
                    n_epoch=EPOCHS,
                    validation_set=(test_x, test_y),
                    show_metric=True,
                    run_id=MODEL_NAME,
                )

                # tensorboard --logdir=foo:D:\Users\Quinten\Desktop\Software_Development\Personal\Python\Rocket-Python\log

                if count % 5 == 0:
                    print("SAVING MODEL!!!")
                    model.save(MODEL_NAME)

        except Exception as e:
            print(str(e))
