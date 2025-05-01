
This project aims to develop an AI agent that can suggest a good placement of tetrominoes by training on replays from top players on [Jstris](https://jstris.jezevec10.com/).

---

## Supervised Learning

Sample size: ~6000 examples. 
Each instance is a tuple (board, piece) where board is a 20x20 matrix and piece is an integer from {1, 2, ..., 7}.
Label is an integer from {1, ... , 48} which represents a unique combination of position and rotation of the falling piece.
I designed three neural network architectures: simple MLP, a CNN, and a multihead attention.
The best performing agent was a simple one layer convolutional neural network.

Loss function: cross-entropy
Optimizer: Adam

### Results
The dataset was divided into 70% training, 20% validation, and 10% test sets.
The max number of epoch was 200, but all networks stopped early due to overfitting.

<p align="center"><img width="464" alt="losses" src="https://github.com/user-attachments/assets/b99849f1-1dde-4397-adb1-30c7248fe603" />

---

## Examples
The following are screenshots of me playing. The ghost pieces are suggestions that the model want me to place the current tetromino.

<p align="center">
  <img width="505" alt="example1" src="https://github.com/user-attachments/assets/767bc4d3-4e0b-4ca1-8686-baf70b16b631" />
  <img width="509" alt="example2" src="https://github.com/user-attachments/assets/2b93162d-7205-46cc-ae18-19d477d130e8" />
</p>

---

## Ideas for Improvement

- Replace binary board encodings with numbers representing tetrominos. (1-7)
- Collect more examples or augment them.
- Make a deeper network.
