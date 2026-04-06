# 🎧 Trigger Word Detection System (Deep Learning)

## 📌 Project Overview

This project implements a **Trigger Word Detection System** using deep learning techniques. The system listens to audio input and detects the presence of a specific keyword:

> **Trigger Word:** `"activate"`

When the trigger word is detected, the system responds by inserting a **chime sound** into the audio output.

---

## 🎯 Objectives

- Build an **end-to-end speech recognition pipeline**
- Generate synthetic training data
- Train a **sequence model (Conv + GRU)**
- Perform real-time keyword detection

---

## 🧠 Core Concept

The system learns a mapping:

$$
\text{Audio Signal} \rightarrow \text{Spectrogram} \rightarrow \text{Neural Network} \rightarrow \text{Trigger Detection}
$$

# ⚙️ System Pipeline

## 1. Audio Input

- Raw waveform sampled at:

  $$
  44,100 \text{ Hz}
  $$

- 10 seconds → 441,000 samples

---

## 2. Feature Extraction (Spectrogram)

Using Short-Time Fourier Transform:

$$
X(t, f) = \sum x[n] \cdot w[n - t] \cdot e^{-j2\pi fn}
$$

### Output Shape:

- $( (101, 5511) $)

---

## 3. Data Synthesis

Each training example:

- Background noise (10 sec)
- 0–4 "activate"
- 0–2 negative words

---

## 4. Label Generation

Convert time (ms) → output index:

$$
t_y = \left\lfloor \frac{t_{ms} \cdot T_y}{10000} \right\rfloor
$$

Set:

$$
y[t_y+1 : t_y+50] = 1
$$

---

## 5. Model Prediction

Output:

$$
\hat{y} \in \mathbb{R}^{1375}
$$

Each value represents probability of trigger word detection

# 🤖 Model Architecture

## Input

$$
X \in \mathbb{R}^{(5511, 101)}
$$

---

## Architecture Breakdown

### 1. Convolution Layer

- Filters: 196
- Kernel size: 15
- Stride: 4

$$
(5511, 101) \rightarrow (1375, 196)
$$

---

### 2. GRU Layers

Two stacked GRU layers:

$$
h_t = \text{GRU}(x_t, h_{t-1})
$$

- Units: 128
- Return sequences: True

---

### 3. Output Layer

TimeDistributed Dense:

$$
\hat{y}_t = \sigma(W h_t + b)
$$

---

## Loss Function

Binary Cross Entropy:

$$
\mathcal{L} = - \frac{1}{T_y} \sum (y \log \hat{y} + (1-y)\log(1-\hat{y}))
$$

# 📁 Project Structure

```text
├── raw_data/
│ ├── activates/
│ ├── negatives/
│ └── backgrounds/
├── audio_examples/
├── XY_train/
├── XY_dev/
├── models/
├── td_utils.py
├── train.wav
├── chime_output.wav
├── model.keras
└── Trigger_Word_Detection.ipynb
```

---

## 📂 Key Components

- **raw_data/** → Audio dataset
- **XY_train/** → Preprocessed training data
- **XY_dev/** → Development set
- **td_utils.py** → Helper functions

# 🚀 Usage Instructions

## 1. Install Dependencies

```bash
pip install numpy pydub matplotlib tensorflow keras
```

## 2. Run Notebook

```bash
jupyter notebook
```

## 3. Train Model

```bash
model.fit(X, Y, batch_size=5, epochs=1)
```

## 4. Evaluate Model

```bash
model.evaluate(X_dev, Y_dev)
```

## 5. Run Prediction

```bash
prediction = detect_triggerword("audio.wav")
chime_on_activate("audio.wav", prediction, threshold=0.5)
```

# 📊 Results & Observations

## Performance

- Accuracy varies depending on training duration
- Short training → poor performance
- Pretrained model → ~94% dev accuracy

---

## ⚠️ Important Note

Accuracy is NOT a reliable metric due to class imbalance:

- Mostly zeros
- Few ones

Better metrics:

- Precision
- Recall
- F1 Score

---

## 🔍 Observations

- Model improves with:
  - More data
  - Better augmentation
  - Longer training

# 🚀 Possible Improvements

## Model Enhancements

- Replace GRU with LSTM
- Add Attention Mechanism
- Use Transformer-based models

---

## Data Improvements

- More diverse accents
- Real-world noisy environments
- Multiple trigger words

---

## Deployment Ideas

- Smart home automation
- Voice assistants
- IoT devices
- Mobile applications

# 🎯 Key Takeaways

- Spectrograms are essential for audio modeling
- Synthetic data enables scalable training
- GRUs capture temporal dependencies
- Label engineering is critical for performance
- Real-time systems require careful design choices

---

## 🧠 Final Insight

This project demonstrates a complete pipeline:

$$
\text{Audio} \rightarrow \text{Signal Processing} \rightarrow \text{Deep Learning} \rightarrow \text{Real-Time Action}
$$

---

## 📌 Conclusion

This is a **production-level concept** used in:

- Voice assistants
- Speech recognition systems
- Wake-word detection engines

Understanding this pipeline provides a strong foundation in:

- Deep Learning
- Audio Processing
- Sequence Modeling
