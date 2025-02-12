# Carlaisle

Carlaisle is an AI that has learned to play Rocket League via Deep Learning.

Carlaisle's process:

[0, 0], GAME START
[0, 0], ROUND START
[1, 0], ROUND END, WIN
[1, 0], ROUND START
[1, 1], ROUND END, LOSS
[1, 1], ROUND START
[2, 1], ROUND END, WIN
[2, 1], GAME END, WIN

Collected data tensor proposed to look like this:

[[[[[frame1, frame2, ..., frameN], WIN], [[frame1, frame2, ..., frameN], LOSS], [[frame1, frame2, ..., frameN], WIN]], WIN],
 [[[[frame1, frame2, ..., frameN], LOSS], [[frame1, frame2, ..., frameN], LOSS], [[frame1, frame2, ..., frameN], WIN]], LOSS]]

This is a list of all games recorded, labeled as a win or a loss. Each of those games are subsected to record each round's win or loss value.
