# Tetris AI Placement Predictor

This project aims to develop an AI agent that can suggest optimal tetromino placements by training on replays from top players on [Jstris](https://jstris.jezevec10.com/).

---

## Setup

* **Sample Size**: \~7000 examples
* **Inputs**:

  * A 20×10 Tetris board represented as an 8×20×12 tensor, where each cell is a one-hot vector of dimension 8. One column is padded on each side for cleaner patch division.
  * A one-hot vector representing the current falling tetromino
* **Outputs**:

  * Logits for column position (0–11)
  * Logits for rotation (0–3)
* **Loss Function**:

  * Sum of cross-entropy loss on both outputs
* **Model Architecture**:

  * Vision Transformer (based on *"An Image is Worth 16x16 Words"*) with custom modifications
* **Dataset Split**:

  * 70% training
  * 20% validation
  * 10% test

<div align="center">
  <img src="https://github.com/user-attachments/assets/3b7b00aa-9bf0-4d06-a44c-17c71e75fa4d" alt="Transformer Architecture">
</div>

---

## Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/4d36eb5e-c60b-4756-9b99-04e2f9366846" alt="Validation Graph 1">
  <img src="https://github.com/user-attachments/assets/a6efa19b-a207-4b97-9a95-052f73b1d7e8" alt="Validation Graph 2">
  <img src="https://github.com/user-attachments/assets/23a61a71-57c1-4340-8aa0-1b857c6b02e4" alt="Validation Graph 3">
</div>

---

## Examples

The following are screenshots of live gameplay. The ghost pieces represent the model's suggested placements for the current tetromino.

<div align="center">
  <img width="505" src="https://github.com/user-attachments/assets/767bc4d3-4e0b-4ca1-8686-baf70b16b631" alt="Gameplay Example 1">
  <img width="509" src="https://github.com/user-attachments/assets/2b93162d-7205-46cc-ae18-19d477d130e8" alt="Gameplay Example 2">
</div>

---

## Ideas for Improvement

* Experiment with simpler network architectures
* Collect more examples or use data augmentation techniques

---
