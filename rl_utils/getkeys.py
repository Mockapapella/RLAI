import keyboard
import mouse
import pygame

pygame.display.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()
buttons = joystick.get_numbuttons()
hats = joystick.get_numhats()
axes = joystick.get_numaxes()

# keyboard_inputs, char, mouse_inputs
keyboard_inputs = ["shift", "alt", "ctrl"]
mouse_inputs = ["left", "right"]

for char in "abcdefghijklmnopqrstuvwxyz 123456789,.'":
    keyboard_inputs.append(char)


def get_inputs():
    inputs_event = []

    for button in keyboard_inputs:
        if keyboard.is_pressed(button):
            inputs_event.append(button)
    for button in mouse_inputs:
        if mouse.is_pressed(button=button):
            inputs_event.append(button)

    # Joystick input for a logitech Gamepad F310
    for i in range(buttons):
        pygame.event.pump()
        button = joystick.get_button(i)
        if i == 2 and button == 1:  # X button pressed
            inputs_event.append("shift")

    for i in range(axes):
        pygame.event.pump()
        axis = joystick.get_axis(i)
        if i == 0 and axis < -0.2:  # Left stick going left
            inputs_event.append("a")
        if i == 0 and axis > 0.2:  # Left stick going right
            inputs_event.append("d")
        if i == 1 and axis > 0.2:  # Left stick going down
            inputs_event.append("s")
        elif i == 2 and axis > 0.15 and "s" not in inputs_event:  # LT is pulled
            inputs_event.append("s")
        if i == 1 and axis < -0.2:  # Left stick going up
            inputs_event.append("w")
        elif i == 2 and axis < -0.15 and "w" not in inputs_event:  # RT is pulled
            inputs_event.append("w")

    # for i in range(hats):
    #     pygame.event.pump()
    #     hat = joystick.get_hat(i)
    #     if i == 0 and hat == (0,1): # Up dpad
    #         inputs_event.append("1")
    #     if i == 0 and hat == (-1,0): # Left dpad
    #         inputs_event.append("2")
    #     if i == 0 and hat == (1,0): # Right dpad
    #         inputs_event.append("3")
    #     if i == 0 and hat == (0,-1): # Left dpad
    #         inputs_event.append("4")

    for i in range(buttons):
        pygame.event.pump()
        button = joystick.get_button(i)
        if i == 0 and button == 1:  # A button pressed
            inputs_event.append("left")
        if i == 1 and button == 1:  # B button pressed
            inputs_event.append("right")

    return inputs_event
