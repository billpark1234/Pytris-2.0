
This project aims to develop an AI agent that can suggest a good placement of tetrominoes by training on replays from top players on [Jstris](https://jstris.jezevec10.com/).

---

## Setup
Sample size: ~7000
Inputs: 20 x 12 tetris board where each cell is a one hot vector of dimension 8 (i.e, 8x20x12 tensor). Also, a one hot vector that represents current falling tetromino.
Output: Logits for position (0-11). Logits for rotation (0-3)
Loss function: sum of cross entropy loss on two outputs.
Model: Vision Transformer (from "image is worth 16 x 16" paper) with some tweaks


## Model Specification
![Transformer drawio(1)](https://github.com/user-attachments/assets/3b7b00aa-9bf0-4d06-a44c-17c71e75fa4d)


## Training configuration
200 epochs
0.1 dropout
32 batch size


### Results
The dataset was divided into 70% training, 20% validation, and 10% test sets.\

![image](https://github.com/user-attachments/assets/b1dd4950-cbcc-4974-8d75-9cd551e50a11)


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
