# RLAI

RLAI is a deep learning approach to playing rocket league. It watches for an active window called "Rocket League", then livestreams the game through a neural network and generates virtual controller presses to control the game.

At a high level, there are 3 steps:

1. Collect training data by recording your own gameplay (`collect.py`)
1. Train the model (`train.py`)
1. Evaluate the model by livestreaming the active Rocket League window to the neural network (`eval.py`)

And additional script is added to playback the training data to verify it's quality (`playback.py`). Note that `opencv` processes images as `BGR` instead of `RGB`, so the two channels will be reversed.

This script was also developed on Ubuntu 24.04 with a PS5 controller, so it is very likely that you will need to perform additional modifications to get this to work on your local system.

# Setup

This repo uses `uv`. This script should get you up and running:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:Mockapapella/RLAI.git
cd RLAI/
uv venv
uv sync
```

# Gathering training data

If you're not interested in gathering training data you can just go to the next section and download the trained model to run it. If how ever you want to collect your own training data, follow these steps:

1. Connect a controller to your computer and boot up Rocket League
1. Run `uv run collect.py` to start collecting labeled training data of (frames, inputs)

It only collects training data when Rocket League is the currently active window. Data is saved to `data/rocket_league/training/` in chunks of 5000 labeled pairs per file, which comes out to about 1GB per file. How quickly this builds up is partially dependent on how fast your computer is. In practice I saw a new file being created every 3-5 minutes. Cancelling the script early will cause it to save the currently gathered training pairs before it fully exits.

# Model and training data

The model and training data are open source. If you either want to train your own model or just inference with the already trained model, you can download them from huggingface:

```
git clone https://huggingface.co/Mockapapella/rlai-1.4M
git clone https://huggingface.co/datasets/Mockapapella/rlai-multi-map
```

From there, to train a model from scratch, run:

```
uv run train.py
```

And to inference with it, boot up rocket league, enter a match (3v3 standard mode is what it was training on), **disconnect the controller you used to collect training data with (otherwise it will create a second controller)**, run the script, and make "Rocket League" the currently active window:

```
uv run eval.py
```

And that should be it!
