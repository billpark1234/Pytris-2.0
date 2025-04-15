
This project aims to develop an AI agent that can suggest a good placement of tetrominoes by training on replays from top players on [Jstris](https://jstris.jezevec10.com/).

---

## Supervised Learning

A convolutional neural network was trained using approximately **6,000 examples** collected from **screenshots of Tetris sprint replays**.

### Results

- Trained for **150 epochs**
- Achieved **~54% accuracy** on the test set
- Significant **overfitting**
- Poor **sample efficiency**

The training results indicate that the model memorizes rather than generalizes. One key limitation is the use of **binary (0/1) board representations**, which discards structural information — such as tetromino combinations that form flat surfaces — potentially harming the model’s ability to recognize important spatial patterns.

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
- Use different neural networks. I used convolutional neural network hoping that it magically captures some patterns in piece placement, but it does not perform as well as expected.
- Collect more examples or augment them.
