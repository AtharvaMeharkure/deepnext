# DeepNext: Advanced Multimodal Deepfake Detection

DeepNext is a powerful, local-first web application designed to detect deepfakes, AI-generated images, and manipulated media with high precision. It uses a custom computer vision feature-extraction pipeline paired with a Multi-Layer Perceptron (MLP) neural network.

---

## 📊 Current Model Performance
- **Model Used:** MLP Classifier (Multi-Layer Perceptron)
- **Cross-Validation Accuracy:** **86.62%**
- **Training Data:** 374 diverse video and image samples (116 Real, 258 Fake) sourced from the Deepfake Detection Challenge (DFDC) dataset.

---

## 🏗️ System Architecture

DeepNext is split into two primary environments to ensure separation of concerns and high performance:

1. **Frontend (User Interface)**
   - **Tech Stack:** React.js, Vite, Framer Motion, Recharts
   - **Functionality:** Provides a stunning, hardware-accelerated user interface. Features interactive confidence rings, dynamic media previewing, real-time pipeline status steps, and interactive feedback loops.
   - **Hosting:** Configured for automated deployment to GitHub Pages via GitHub Actions.

2. **Backend (Inference Engine)**
   - **Tech Stack:** FastAPI (Python), OpenCV, MediaPipe, Scikit-Learn, SQLite
   - **Functionality:** Handles heavy video processing, extracts 545 distinct computer-vision features per file, runs inference via `.joblib` saved neural networks, and maintains the local User Correction Memory Bank.
   - **Hosting:** Runs locally on `http://127.0.0.1:8000` (Must be deployed to a Python-capable cloud provider like Render or Railway for public access).

---

## 🧠 How It Works: The Logic & Pipeline

When a user uploads a video or image, the file undergoes a rigorous 5-step analysis pipeline:

### 1. Frame Extraction & Preprocessing
If the uploaded file is a video, OpenCV extracts up to **30 evenly spaced frames** across the duration of the clip. If it is an image, it is treated as a single frame.

### 2. Facial Landmark Meshing (MediaPipe)
Google's MediaPipe Face Mesh is applied to each frame. This maps out **478 3D landmarks** across the face, identifying the exact coordinates of the eyes, lips, jawline, and nose.

### 3. Feature Extraction (The 545-Dimension Vector)
Instead of relying on a "black box" deep learning model (like a CNN) that might overfit, DeepNext extracts explicit, mathematically quantifiable features that are known to fail in AI-generated deepfakes. It generates a **545-dimensional feature vector** analyzing:
- **Face Geometry:** Calculates the distances and angles between facial landmarks to spot anatomical impossibilities (e.g., eyes too far apart in specific frames).
- **Bilateral Symmetry:** Measures the asymmetry between the left and right sides of the face. AI generators frequently struggle to maintain perfect structural symmetry.
- **Frequency Domain (DCT):** Converts the image into the frequency domain to look for high-frequency noise and grid-like artifacts left behind by Generative Adversarial Networks (GANs).
- **Optical Flow:** (Videos only) Analyzes the motion vectors between consecutive frames. Deepfakes often exhibit temporal flickering or unnatural morphing between frames.
- **Texture Consistency:** Checks the variance in skin textures to identify blurring or smoothing often applied by deepfake face-swapping algorithms.

### 4. Machine Learning Inference
The 545 features are aggregated, normalized using a Standard Scaler, and fed into the active ML model (currently an **MLP Classifier**). The neural network calculates the probability of the media being fake.
- If the Fake Probability is `>= 0.50`, it is flagged as **FAKE**.
- Otherwise, it is labeled **REAL**.

### 5. The Feedback Loop (Memory Bank & Retraining)
No AI is perfect. If the model predicts an image incorrectly, the user can click "Actually it's REAL" or "Actually it's FAKE" in the UI. 
- **Instant Correction:** The exact feature vector of that image is instantly saved to a local SQLite database (`store/features.db`).
- **Memory Bank (1-NN):** The next time that *exact* media is processed, the backend intercepts the neural network, uses a 1-Nearest Neighbor search to recognize the saved feature vector, and instantly returns the user's manual correction with 100% confidence.
- **Batch Retraining:** Once the user submits enough manual corrections (10+), the backend unlocks and performs a full background retraining of the neural network to permanently integrate the new edge cases into its logic.