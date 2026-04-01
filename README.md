# Face-Recognition-Algorithms
![Face Recognition_ From Classical Algorithms to Deep Learning_Difference between classical and modern, don&#39;t write anything just picture](https://github.com/user-attachments/assets/cb4b3f3c-72bb-4716-9a6c-8b4357ffde77)

## This project presents a comprehensive comparison of multiple face recognition approaches, ranging from classical algorithms to modern deep learning.

The goal is to analyze how face recognition systems evolved from handcrafted feature extraction to deep representation learning and metric learning.

---

## 🚀 Implemented Methods

### 🔹 Classical Approaches
- Eigenfaces (PCA)
- Fisherfaces (LDA)
- LBPH (Local Binary Patterns Histograms)
- HOG + SVM

### 🔹 Deep Learning & Similarity-Based Models
- Siamese Network (Contrastive Loss)
- FaceNet (Triplet Loss)
- DeepFace (Classification-based)

### 🔹 Classical Approaches

#### 1. Eigenfaces (PCA)
Eigenfaces is one of the earliest face recognition techniques. It projects face images into a lower-dimensional space using Principal Component Analysis (PCA).

- Learns a set of orthogonal components (eigenvectors) representing face features
- Reduces dimensionality while preserving maximum variance
- Classification is typically done using nearest neighbor
![eigen](https://github.com/user-attachments/assets/d158e9e6-0810-4124-9cd0-b5951af9504d)

**Pros:**
- Simple and fast
- Works well in controlled environments

**Cons:**
- Sensitive to lighting and pose variations
- Needs retraining when new identities are added
- Limited accuracy in real-world scenarios

---

#### 2. Fisherfaces (LDA)
Fisherfaces improves upon Eigenfaces by using Linear Discriminant Analysis (LDA), focusing on class separability.

- Maximizes between-class variance and minimizes within-class variance
- More robust to lighting and expression changes
![fisher1-1](https://github.com/user-attachments/assets/b2436e60-ad09-40bc-8067-7f4cfe6fb18e)

**Pros:**
- Better discrimination than PCA
- More robust to illumination variations

**Cons:**
- Higher computational cost
- Less compact representation compared to PCA

---

#### 3. LBPH (Local Binary Patterns Histograms)
LBPH is a texture-based method that encodes local patterns of pixels.

- Converts local neighborhoods into binary patterns
- Uses histograms to represent facial features
<img width="1600" height="403" alt="LBPH1" src="https://github.com/user-attachments/assets/006a6a47-97d4-4ded-b058-d3a80e38482a" />

**Pros:**
- Simple and efficient
- Works well in controlled environments
- Robust to grayscale changes

**Cons:**
- Limited performance in complex scenarios
- Sensitive to noise and occlusion

---

#### 4. HOG + SVM
This approach combines feature extraction and classification:

- HOG (Histogram of Oriented Gradients): captures edge and gradient structure
- SVM (Support Vector Machine): classifies faces using a separating hyperplane
<img width="1544" height="307" alt="HoG2SVM" src="https://github.com/user-attachments/assets/9b345be6-bd8f-48c1-b8ca-1f864679d6e1" />

**Pros:**
- Good performance for structured features
- Robust to small variations in pose and lighting

**Cons:**
- Depends on handcrafted features
- Limited scalability compared to deep learning

---

## 🔗 Similarity Learning (Metric Learning)

Similarity learning focuses on learning embeddings where similar faces are closer and different faces are farther apart.

### 5. Siamese Networks
A Siamese Network consists of two identical neural networks with shared weights.

- Takes pairs of images (positive/negative)
- Learns to compute similarity between them
<img width="1200" height="803" alt="siamese" src="https://github.com/user-attachments/assets/c00fdff6-c9c6-42b2-b76d-71616d000e0b" />


**Key Concept:**
- Embedding learning instead of direct classification

---

### Contrastive Loss
Used to train Siamese Networks:

- Minimizes distance between similar pairs
- Maximizes distance between dissimilar pairs

---

## 🤖 Deep Learning Approaches

### 6. FaceNet
FaceNet learns a compact embedding for each face.

- Produces 128-dimensional feature vectors
- Uses Triplet Loss for training
<img width="1272" height="249" alt="facenet_1" src="https://github.com/user-attachments/assets/94cb680d-69c8-4bd3-821c-cda8943037c0" />

**Triplet Components:**
- Anchor (A)
- Positive (P)
- Negative (N)
<img width="333" height="88" alt="triplet-loss" src="https://github.com/user-attachments/assets/407d664e-ac46-48ce-93d5-8d4ae2e9b694" />

**Objective:**
Ensure:


**Pros:**
- High accuracy
- Scalable and flexible
- Works well for verification and identification

---

### 7. DeepFace
DeepFace is a deep CNN-based model and classification

- Trained on face datasets
- Primarily classification-based

**Key Features:**
- Uses deep neural networks for feature extraction
- Can also be used to generate embeddings

**Pros:**
- High performance
- Strong baseline for deep learning

**Cons:**
- Less flexible than embedding-based methods
- Computationally expensive

---

## 📊 Results

| Model            |Accuracy|
|------------------|--------|
| Eigenfaces       | 66.67% |
| Fisherfaces      | 70.37% |
| LBPH             | 55.56% |
| HOG + SVM        | 70.37% |
| Siamese Network  | 84.61% |
| FaceNet          | 90.90% |
| DeepFace         | 85.18% |

---

## 🧠 Key Insights

- Classical methods rely on handcrafted features and struggle with real-world variations.
- Deep learning significantly improves performance by learning features automatically.
- Similarity learning enables better generalization for unseen identities.
- FaceNet achieved the best performance due to effective embedding learning.
- Transitioning from classification to metric learning is a key breakthrough in face recognition.

---

## 🎯 Conclusion

This project demonstrates the evolution of face recognition:

**Feature Engineering → Deep Learning → Metric Learning**

Each stage improves robustness, scalability, and real-world performance.

---

## 🚧 Future Work

- Implement ArcFace and CosFace (margin-based losses)
- Evaluate on larger datasets (LFW, VGGFace2)
- Improve performance under real-world conditions
- Deploy real-time face recognition system

---

## 👨‍💻 Author

Ahmed Hussien  
Computer Vision Engineer | Machine Learning Enthusiast

