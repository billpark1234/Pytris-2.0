I trained a neural network using ~6000 examples collected from screenshots of replays of top players in Jstris. 
With 150 epochs of training, the agent achieved about 45% of test set accuracy. The learning algorithm seems to overfit and has a poor sample efficiency.


As for the reinforcement learning agent, training hasn't started yet. Next big task is to figure out the right reward function. Another person already
came up with a reward function that approximates the fitness of a tetris board, using genetic algorithm. However, it still seems to play like a machine, not like a
human being. By playing like a machine, I mean that current AIs out there don't consider finnesse. They ignore key mechanics like DAS and wallkicks which are necessary
for efficient block placement. I am not sure whether a proper reward function will motivate agents to play like human or a fundamentally different approach is required.


An imporovment might be possible by not converting the boards into 0-1 arrays. Tetrominoes are often used in combination to form rectangles without bumps,
and such information is lost when the blocks are converted to 0s and 1s.
