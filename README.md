# LL784-Soli-Gesture-Recognition

# Fast Time-as-Channel LSGAN + DANN for Soli Gesture Recognition

[[OpenInColab([https://colab.research.google.com/assets/colabbadge.svg(https://colab.research.google.com/drive/1vmXbLK2_Q5WW1dmJ9LRYipOpD3n8g4wl?authuser=1#scrollTo=_akxFd9aj3CM))]

This repository contains the code for Assignment 3 (ELL784) at IIT Delhi. It implements a Fast Time-as-Channel Least Squares Generative Adversarial Network (LSGAN) combined with a Domain-Adversarial Neural Network (DANN) for fine-grained classification of Soli radar data.

## 🧠 Model Architecture

The solution is designed to handle temporal radar data efficiently while remaining invariant to different subjects, utilizing three main components:

1. **Feature Extractor & Classifier (3D CNN + LSTM)**
   * **Spatial Processing:** A 4-layer 3D Convolutional Neural Network extracts spatial features from the Range-Doppler maps while preserving the temporal dimension ($T=40$).
   * **Temporal Modeling:** An LSTM network processes the sequential features, and the final hidden state is passed to a dense layer to classify the 11 gestures.

2. **Data Augmentation (Fast LSGAN)**
   * To address data scarcity in fine-grained classes, a Fast LSGAN is utilized.
   * **Generator:** Treats "Time" as the channel dimension, generating all 40 frames simultaneously using `ConvTranspose2d` layers for massive speedups.
   * **Discriminator:** Evaluates the realism of the generated 40-frame sequence using Spectral Normalization for stability.

3. **Subject Invariance (DANN)**
   * A Domain Classifier is attached to the feature extractor via a **Gradient Reversal Layer (GRL)**.
   * It attempts to predict the Subject ID (1 to 16). The GRL forces the feature extractor to learn representations that are predictive of the gesture but completely agnostic to the specific subject performing it.

## 📦 Final Model Weights
The fully trained model checkpoints (containing the feature extractor, classifier, and generator weights) can be downloaded here:
🔗 **[Download Final Trained Model (.pt) Here](INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE)**

## 🚀 Instructions to Run

### 1. Data Preparation
Extract the dataset containing the `.h5` files. Open `main.py` and update the `DATA_DIR` variable to point to your extracted directory:
```python
DATA_DIR = '/path/to/extracted_files/dsp'

====================================================================
                     PART 1: DATA AUGMENTATION (LSGAN)
====================================================================

      [ Latent Vector (Z) ] + [ Target Gesture Label ]
                  |                      |
                  +----------+-----------+
                             |
                             v
                 +------------------------+
                 |  Fast LSGAN Generator  | 
                 |   (Time-as-Channel)    |
                 +------------------------+
                             |
                             v
                  [ Synthetic Radar Data ] 
                    (T=40, R=32, D=32)
                             |
                             v
                 +------------------------+
 [ Real Data ] ->|  LSGAN Discriminator   | -> [ Real or Fake? ]
                 +------------------------+

====================================================================
           PART 2: GESTURE CLASSIFICATION & SUBJECT INVARIANCE 
====================================================================

           [ Real Radar Data ] or [ Synthetic Radar Data ]
                                 |
                                 v
                     +-----------------------+
                     |   Feature Extractor   | 
                     |    (4-Layer 3D CNN)   |
                     +-----------------------+
                                 |
           +---------------------+---------------------+
           |                                           |
           v                                           v
 +-------------------+                       +-------------------+
 |   LSTM Network    |                       | Gradient Reversal |
 |(Temporal Modeling)|                       |    Layer (GRL)    |
 +-------------------+                       +-------------------+
           |                                           |
           v                                           v
 +-------------------+                       +-------------------+
 | Gesture Classifier|                       | Domain Classifier |
 |   (Dense Layers)  |                       |   (Dense Layers)  |
 +-------------------+                       +-------------------+
           |                                           |
           v                                           v
 [ Predicted Gesture ]                       [ Predicted Subject ]
     (11 Classes)                           (Forces Subject Invariance)
