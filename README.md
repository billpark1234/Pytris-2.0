# Tetris AI: Learning from Jstris Replays

This project aims to develop an AI agent that can learn to play Tetris at a high level by training on replays from top players on [Jstris](https://jstris.jezevec10.com/). The project combines supervised learning and reinforcement learning to eventually create a model that plays not only effectively, but also in a way that resembles human gameplay.

---

## 🧠 Supervised Learning

A neural network was trained using approximately **6,000 examples** collected from **screenshots of expert replays**.

### Results

- Trained for **150 epochs**
- Achieved **~45% accuracy** on the test set
- Significant **overfitting**
- Poor **sample efficiency**

The training results indicate that the model memorizes rather than generalizes. One key limitation is the use of **binary (0/1) board representations**, which discards structural information — such as tetromino combinations that form flat surfaces — potentially harming the model’s ability to recognize important spatial patterns.

<p align="center">
  <img width="460" alt="Training Loss" src="https://github.com/user-attachments/assets/b5a96eb8-bfec-4df6-950e-8a0cc6904558" />
</p>

---

## 🤖 Reinforcement Learning (Upcoming Work)

The next major goal is to train a reinforcement learning (RL) agent. A reward function already exists — designed using a **genetic algorithm** to approximate the "fitness" of a Tetris board — but it leads to robotic gameplay.

### Limitations of Current AI Behavior

- Ignores **finesse** (minimal keystroke efficiency)
- Does not account for **DAS (Delayed Auto Shift)** and **wallkicks**
- Lacks timing and input restrictions that humans naturally face

Current agents play in a "mechanical" way, focusing solely on board state without regard for human-like constraints or inputs. This makes them unrealistic and sometimes inefficient when translated into actual key sequences.

---

## 🧩 Example Inputs

The following are samples of board states used in training:

<p align="center">
  <img width="505" alt="example1" src="https://github.com/user-attachments/assets/767bc4d3-4e0b-4ca1-8686-baf70b16b631" />
  <img width="509" alt="example2" src="https://github.com/user-attachments/assets/2b93162d-7205-46cc-ae18-19d477d130e8" />
</p>

---

## 🛠️ Ideas for Improvement

- Replace binary board encodings with **structured inputs**, such as per-cell tetromino identity or surface maps.
- Incorporate **timing constraints**, including DAS and ARR modeling.
- Introduce **finesse metrics** into the reward function.
- Use **imitation learning** or **inverse reinforcement learning** to better reflect human-like playstyles.
- Model **input legality**, e.g., limiting movements to those actually possible within a fixed number of frames.

---

## 📌 Goals

- Train a model that **learns from top players**
- Develop an RL agent that is **efficient**, **realistic**, and **human-like**
- Push the boundary of AI Tetris beyond current "robotic" systems

---

## 📁 Status

- [x] Collected expert data  
- [x] Trained supervised model  
- [ ] Designed reward function  
- [ ] Trained reinforcement learning agent  

---

Stay tuned as this project evolves to explore new ways to bridge the gap between high-level human strategy and machine learning.
