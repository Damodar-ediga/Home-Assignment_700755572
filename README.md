# Home Assignment 3

**Name:** Damodar Goud Ediga  
**ID:** 700755572  

---

## Assignment Overview
This repository contains the implementations for four deep learning tasks using TensorFlow/Keras:
1. **Basic Autoencoder for Image Reconstruction**
2. **Denoising Autoencoder**
3. **RNN for Text Generation**
4. **Sentiment Classification Using LSTM**

Each task is implemented in Python with TensorFlow/Keras and follows the given submission requirements.

---

## Q1: Implementing a Basic Autoencoder
### **Objective:**
Train a fully connected autoencoder on the MNIST dataset to reconstruct input images.

### **Implementation Steps:**
1. Load the MNIST dataset and normalize the pixel values.
2. Define an encoder with a dense layer (32 units) and a decoder with an output layer (784 units).
3. Compile and train the model using binary cross-entropy loss.
4. Evaluate the autoencoder by comparing original vs. reconstructed images.
5. Modify the latent dimension size and analyze reconstruction quality.

### **Files:**
- `autoencoder.py` (contains model implementation)
- `results/autoencoder_output.png` (sample output images)

---

## Q2: Implementing a Denoising Autoencoder
### **Objective:**
Train an autoencoder to reconstruct clean images from noisy inputs.

### **Implementation Steps:**
1. Modify the basic autoencoder by adding Gaussian noise (mean=0, std=0.5) to input images.
2. Train the model while keeping the clean image as the target.
3. Compare the performance of the basic autoencoder vs. denoising autoencoder.
4. Visualize noisy vs. denoised images.
5. Explain a real-world application (medical imaging).

### **Files:**
- `denoising_autoencoder.py`
- `results/denoising_output.png`

---

## Q3: Implementing an RNN for Text Generation
### **Objective:**
Train an LSTM-based RNN to generate text sequences character by character.

### **Implementation Steps:**
1. Load a text dataset (e.g., Shakespeare's Sonnets).
2. Convert text into a sequence of character indices (one-hot encoding or embeddings).
3. Define an RNN model using LSTM layers.
4. Train the model and generate text using temperature scaling.
5. Explain the effect of temperature scaling on randomness.

### **Files:**
- `text_generation.py`
- `results/generated_text.txt`

---

## Q4: Sentiment Classification Using RNN
### **Objective:**
Train an LSTM-based model for sentiment classification on the IMDB dataset.

### **Implementation Steps:**
1. Load the IMDB dataset and preprocess text (tokenization and padding).
2. Build an LSTM-based binary classification model.
3. Train the model to classify reviews as positive or negative.
4. Evaluate using a confusion matrix and classification report.
5. Explain the importance of precision-recall tradeoff.

### **Files:**
- `sentiment_analysis.py`
- `results/confusion_matrix.png`

---

## Conclusion
This project demonstrates autoencoder-based image reconstruction and denoising, as well as RNN-based text generation and sentiment classification. Each task highlights key concepts in deep learning and sequence modeling.

---

**Author:** Damodar Goud Ediga  
**ID:** 700755572

