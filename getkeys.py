import keyboard
import mouse

available_inputs = ["shift", "alt", "ctrl"]
available_inputs_2 = ["left", "right"]

for char in "abcdefghijklmnopqrstuvwxyz 123456789,.'":
    available_inputs.append(char)

def key_mouse_check():
    inputs_event = []
    for button in available_inputs:
        if keyboard.is_pressed(button):
            inputs_event.append(button)
    for button_prime in available_inputs_2:
        if mouse.is_pressed(button=button_prime):
            inputs_event.append(button_prime)

    return inputs_event

print(available_inputs)