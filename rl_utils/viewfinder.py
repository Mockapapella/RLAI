import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


PATH = "./custom_mnist_results/model.pth"

# Initialize Model
CUDA_GPU_TO_USE = 1
DEVICE = torch.device("cuda:{}".format(CUDA_GPU_TO_USE) if torch.cuda.is_available() else "cpu")
network = MnistNet()
network.load_state_dict(torch.load(PATH))
network.to(DEVICE)
print(network.load_state_dict(torch.load(PATH)))

SCREEN_CAPTURE_LEFT_MY_TEAM = 545
SCREEN_CAPTURE_TOP_MY_TEAM = 50
SCREEN_CAPTURE_RIGHT_MY_TEAM = 42
SCREEN_CAPTURE_BOTTOM_MY_TEAM = 45
SCREEN_CROP_MY_TEAM = {
    "top": SCREEN_CAPTURE_TOP_MY_TEAM,
    "left": SCREEN_CAPTURE_LEFT_MY_TEAM,
    "width": SCREEN_CAPTURE_RIGHT_MY_TEAM,
    "height": SCREEN_CAPTURE_BOTTOM_MY_TEAM,
}

SCREEN_CAPTURE_LEFT_THEIR_TEAM = 730
SCREEN_CAPTURE_TOP_THEIR_TEAM = 50
SCREEN_CAPTURE_RIGHT_THEIR_TEAM = 42
SCREEN_CAPTURE_BOTTOM_THEIR_TEAM = 45
SCREEN_CROP_THEIR_TEAM = {
    "top": SCREEN_CAPTURE_TOP_THEIR_TEAM,
    "left": SCREEN_CAPTURE_LEFT_THEIR_TEAM,
    "width": SCREEN_CAPTURE_RIGHT_THEIR_TEAM,
    "height": SCREEN_CAPTURE_BOTTOM_THEIR_TEAM,
}


def display_score(score, DEVICE, mnist=False):
    sct = mss.mss()
    screen = np.asarray(sct.grab(score))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _, screen = cv2.threshold(screen, 200, 255, cv2.THRESH_BINARY)
    screen = cv2.dilate(screen, (3, 3), iterations=3)
    screen = cv2.resize(screen, (28, 28))
    cv2.imshow("Captured View {}".format(str(score)), screen)

    if mnist is True:
        screen_tensor = torch.tensor(screen, dtype=torch.float).to(DEVICE)
        screen_tensor = screen_tensor[None, None].to(DEVICE)
        screen_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = network(screen_tensor).to(DEVICE)
            _, prediction = torch.max(outputs.data, 1)
            print(prediction)


def mnist_detection(SCREEN_CROP_MY_TEAM, SCREEN_CROP_THEIR_TEAM, DEVICE):
    while True:
        display_score(SCREEN_CROP_MY_TEAM, DEVICE, mnist=True)
        display_score(SCREEN_CROP_THEIR_TEAM, DEVICE, mnist=True)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


def screen_viewfinder(SCREEN_CROP_MY_TEAM, SCREEN_CROP_THEIR_TEAM):

    while True:
        display_score(SCREEN_CROP_MY_TEAM)
        display_score(SCREEN_CROP_THEIR_TEAM)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # screen_viewfinder(SCREEN_CROP_MY_TEAM, SCREEN_CROP_THEIR_TEAM)
    mnist_detection(SCREEN_CROP_MY_TEAM, SCREEN_CROP_THEIR_TEAM, DEVICE)
