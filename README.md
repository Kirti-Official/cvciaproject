# Deep Learning for Diabetic Retinopathy Screening from Retinal Fundus Images

A computer vision and deep learning system for **five-class diabetic retinopathy (DR) classification** from retinal fundus images using a pretrained **EfficientNet-B0** backbone enhanced with **CBAM (Convolutional Block Attention Module)**. The project also integrates **Grad-CAM** for visual explainability and provides an interactive **Streamlit dashboard** for inference.

> **Project Status:** Research / academic prototype
> **Current Dataset:** APTOS 2019 Blindness Detection
> **Current Task:** Five-class diabetic retinopathy severity classification

---

## 📌 Overview

Diabetic retinopathy is a diabetes-related retinal condition that can progress through different stages of severity. Automated analysis of retinal fundus images can assist in screening by identifying visual patterns associated with different stages of DR.

This project develops an end-to-end computer vision pipeline that takes a retinal fundus image as input and predicts one of five DR severity levels:

| Class | Severity                           |
| ----: | ---------------------------------- |
|     0 | No Diabetic Retinopathy            |
|     1 | Mild Diabetic Retinopathy          |
|     2 | Moderate Diabetic Retinopathy      |
|     3 | Severe Diabetic Retinopathy        |
|     4 | Proliferative Diabetic Retinopathy |

The implementation focuses on combining:

* Retinal image preprocessing
* Contrast enhancement using CLAHE
* Transfer learning
* EfficientNet-B0
* CBAM channel and spatial attention
* Class-weighted training
* Multiple evaluation metrics
* Grad-CAM explainability
* Interactive Streamlit inference

---

## 🎯 Objectives

The main objectives of the project are:

1. Develop a preprocessing pipeline for retinal fundus images.
2. Build an EfficientNet-B0 baseline for five-class DR classification.
3. Improve the feature representation using CBAM attention.
4. Handle class imbalance using class-weighted loss.
5. Compare the baseline and attention-enhanced models.
6. Provide visual explanations using Grad-CAM.
7. Deploy the trained model through an interactive Streamlit dashboard.

---

## 🧠 System Architecture

```text
                 Retinal Fundus Image
                          │
                          ▼
                 Image Preprocessing
                          │
            ┌─────────────┴─────────────┐
            │                           │
       Retinal Crop                 CLAHE
            │                           │
            └─────────────┬─────────────┘
                          ▼
                 Resize to 224 × 224
                          │
                          ▼
                    Normalization
                          │
                          ▼
                  EfficientNet-B0
                  Feature Extractor
                          │
                          ▼
                   Feature Map
                    1280 × 7 × 7
                          │
                          ▼
                        CBAM
              ┌───────────┴───────────┐
              │                       │
       Channel Attention       Spatial Attention
              │                       │
              └───────────┬───────────┘
                          ▼
                Global Average Pooling
                          │
                          ▼
                       Dropout
                          │
                          ▼
                  Fully Connected Layer
                       1280 → 5
                          │
                          ▼
                  DR Classification
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           No DR        Mild       Moderate
             │
             ├──────── Severe
             │
             └──────── Proliferative

                          │
                          ▼
                       Grad-CAM
                          │
                          ▼
                 Explainable Output
                          │
                          ▼
                  Streamlit Dashboard
```

---

## 📊 Dataset

The current implementation uses the **APTOS 2019 Blindness Detection** dataset.

### Dataset size

**3,662 retinal fundus images**

### Class distribution

|     Class | Description      |    Images |
| --------: | ---------------- | --------: |
|         0 | No DR            |     1,805 |
|         1 | Mild DR          |       370 |
|         2 | Moderate DR      |       999 |
|         3 | Severe DR        |       193 |
|         4 | Proliferative DR |       295 |
| **Total** |                  | **3,662** |

The dataset is imbalanced, with substantially fewer samples in some disease stages.

For this reason, the training process uses **class-weighted Cross Entropy Loss**.

---

## 🔀 Dataset Split

A stratified 80/20 split is used.

### Training set

**2,929 images**

| Class | Images |
| ----: | -----: |
|     0 |  1,444 |
|     1 |    296 |
|     2 |    799 |
|     3 |    154 |
|     4 |    236 |

### Validation set

**733 images**

| Class | Images |
| ----: | -----: |
|     0 |    361 |
|     1 |     74 |
|     2 |    200 |
|     3 |     39 |
|     4 |     59 |

Stratification is used to maintain approximately similar class proportions across training and validation sets.

---

# 🔧 Preprocessing

Retinal fundus images can contain unnecessary black background, uneven illumination and varying contrast.

The implemented preprocessing pipeline is:

```text
Original Image
      │
      ▼
Retinal Region Detection
      │
      ▼
Background Cropping
      │
      ▼
Aspect-Ratio-Preserving Resize
      │
      ▼
Black Padding
      │
      ▼
CLAHE Contrast Enhancement
      │
      ▼
224 × 224 Image
      │
      ▼
ImageNet Normalization
```

### 1. Retinal cropping

The preprocessing pipeline identifies the non-black retinal region and removes unnecessary background.

### 2. Aspect-ratio-preserving resize

Images are resized while preserving their original aspect ratio. Black padding is then used to obtain the required 224 × 224 input size.

### 3. CLAHE

CLAHE stands for **Contrast Limited Adaptive Histogram Equalization**.

It is applied to the luminance component in LAB color space.

Configuration:

```text
clipLimit = 2.0
tileGridSize = (8, 8)
```

CLAHE improves local contrast while limiting excessive amplification of noise.

### 4. Normalization

The processed image is normalized using ImageNet-style channel statistics to match the pretrained EfficientNet input distribution.

---

# 🔄 Data Augmentation

Training images use:

* Random horizontal flip — probability 0.5
* Random rotation — ±10°
* Color jitter

  * Brightness = 0.15
  * Contrast = 0.15

Validation images do not use random augmentation.

The purpose of augmentation is to increase training diversity and reduce overfitting.

---

# 🏗️ Model Architecture

## EfficientNet-B0 Baseline

The first model establishes a baseline using a pretrained EfficientNet-B0.

```text
Input Image
     │
     ▼
EfficientNet-B0
     │
     ▼
Feature Extraction
     │
     ▼
Global Pooling
     │
     ▼
Classifier
     │
     ▼
5 DR Classes
```

The final classification layer is adapted to output five classes.

### Baseline parameters

Approximately:

**4.01 million parameters**

---

# 👁️ CBAM Attention

The second model extends EfficientNet-B0 with **CBAM — Convolutional Block Attention Module**.

CBAM applies two forms of attention sequentially:

```text
Feature Map
     │
     ▼
Channel Attention
     │
     ▼
Spatial Attention
     │
     ▼
Refined Feature Map
```

## Channel Attention

Channel attention determines **which feature channels are important**.

Average pooling and maximum pooling are used to generate channel descriptors, followed by a shared lightweight MLP and sigmoid activation.

The implementation uses a reduction ratio of:

```text
16
```

## Spatial Attention

Spatial attention determines **where important information is located**.

Average-pooled and max-pooled feature maps are concatenated and processed using a 7 × 7 convolution followed by sigmoid activation.

### CBAM model parameters

Approximately:

**4.22 million parameters**

Additional parameters compared with the baseline:

**204,898**

The CBAM module therefore adds relatively little overhead compared with replacing the entire backbone.

---

# ⚖️ Class-Weighted Loss

Because the dataset is imbalanced, class weights are calculated from the training distribution.

The approximate weights used are:

| Class | Weight |
| ----: | -----: |
|     0 | 0.4057 |
|     1 | 1.9791 |
|     2 | 0.7332 |
|     3 | 3.8039 |
|     4 | 2.4822 |

The loss gives higher importance to minority classes.

For example, the severe DR class receives a substantially higher weight than the majority no-DR class.

This helps reduce bias toward the majority class.

---

# ⚙️ Training Configuration

| Parameter         | Value                        |
| ----------------- | ---------------------------- |
| Input size        | 224 × 224                    |
| Batch size        | 8                            |
| Optimizer         | AdamW                        |
| Learning rate     | 3e-4                         |
| Weight decay      | 1e-4                         |
| Training epochs   | 5                            |
| Loss              | Class-weighted Cross Entropy |
| Number of classes | 5                            |
| Backbone          | EfficientNet-B0              |
| Attention         | CBAM                         |
| Device            | CPU                          |

The current experiment was performed in a CPU-only environment with limited computational resources.

---

# 📈 Evaluation Metrics

The models are evaluated using multiple metrics.

## Accuracy

Measures the overall proportion of correct predictions.

$$
Accuracy =
\frac{Correct\ Predictions}{Total\ Predictions}
$$

## Precision

Measures how many predictions assigned to a class are actually correct.

$$
Precision =
\frac{TP}{TP+FP}
$$

## Recall

Measures how many actual samples of a class are correctly identified.

$$
Recall =
\frac{TP}{TP+FN}
$$

## F1 Score

The harmonic mean of precision and recall.

$$
F1 =
\frac{2PR}{P+R}
$$

## Macro F1

F1 is calculated independently for each class and then averaged.

Macro F1 is particularly useful for this project because the dataset is imbalanced.

## Quadratic Weighted Kappa

QWK measures agreement while considering the ordinal relationship between DR severity classes.

This is useful because:

```text
No DR → Mild → Moderate → Severe → Proliferative
```

is an ordered progression.

---

# 🧪 Experimental Results

Two models were evaluated:

1. EfficientNet-B0 baseline
2. EfficientNet-B0 + CBAM

### Overall comparison

| Metric              | EfficientNet-B0 | EfficientNet-B0 + CBAM |
| ------------------- | --------------: | ---------------------: |
| Validation Accuracy |          77.08% |             **78.99%** |
| Macro F1            |          60.30% |             **60.52%** |
| QWK                 |      **85.57%** |                 85.46% |
| Parameters          |           4.01M |                  4.22M |

### Interpretation

The CBAM model improved validation accuracy by approximately:

**+1.91 percentage points**

Macro F1 increased by approximately:

**+0.22 percentage points**

The improvement is therefore **modest rather than dramatic**.

QWK was very similar between the two selected checkpoints.

---

# 📋 CBAM Per-Class Results

| Class         | Precision | Recall |     F1 |
| ------------- | --------: | -----: | -----: |
| No DR         |    97.19% | 95.84% | 96.51% |
| Mild          |    45.61% | 70.27% | 55.32% |
| Moderate      |    78.21% | 70.00% | 73.88% |
| Severe        |    40.00% | 15.38% | 22.22% |
| Proliferative |    50.72% | 59.32% | 54.69% |

The strongest performance is observed for the **No DR** class.

The **Severe DR** class remains challenging because it has relatively few training examples and is frequently confused with neighboring severity classes.

---

# 🔍 Confusion Matrix Analysis

The CBAM confusion matrix is:

```text
[[346, 13,  2,  0,  0],
 [  9, 52, 11,  0,  2],
 [  1, 36,140,  7, 16],
 [  0,  1, 16,  6, 16],
 [  0, 12, 10,  2, 35]]
```

The major errors occur between neighboring severity classes, particularly:

```text
Mild ↔ Moderate
Moderate ↔ Severe
Severe ↔ Proliferative
```

This reflects the difficulty of distinguishing visually similar stages of diabetic retinopathy.

---

# 🔬 Grad-CAM Explainability

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations of predictions.

The process is:

```text
Input Image
     │
     ▼
Model Prediction
     │
     ▼
Selected Class
     │
     ▼
Gradient Calculation
     │
     ▼
Feature Activations
     │
     ▼
Weighted Feature Maps
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Overlay on Original Image
```

Grad-CAM highlights image regions that contributed strongly to the selected prediction.

### Important limitation

Grad-CAM is an **explainability method**, not a lesion detector.

It does not explicitly:

* identify individual lesions
* produce lesion bounding boxes
* segment lesions
* provide clinical lesion measurements

It provides visual evidence about where the model's decision was influenced.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows a user to:

1. Upload a retinal fundus image.
2. View the original image.
3. View the preprocessed image.
4. Run the trained EfficientNet-B0 + CBAM model.
5. View the predicted DR class.
6. View class probabilities.
7. Generate a Grad-CAM heatmap.
8. View the Grad-CAM overlay.

### Dashboard flow

```text
Upload Image
     │
     ▼
Preprocessing
     │
     ▼
CBAM Model
     │
     ▼
Prediction
     │
     ├── Predicted Class
     │
     ├── Class Probabilities
     │
     └── Grad-CAM
```

---

# 📁 Project Structure

```text
DR_CV_Project/
│
├── .gitignore
│
├── src/
│   ├── cache_preprocessing.py
│   ├── cbam.py
│   ├── cbam_model.py
│   ├── dashboard.py
│   ├── dataset.py
│   ├── gradcam.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── split_dataset.py
│   ├── train_baseline.py
│   ├── train_cbam.py
│   └── visualize_dataset.py
│
└── outputs/
    └── plots/
        ├── baseline_accuracy_curve.png
        ├── baseline_confusion_matrix.png
        ├── baseline_loss_curve.png
        ├── cbam_accuracy_curve.png
        ├── cbam_confusion_matrix.png
        ├── cbam_loss_curve.png
        ├── class_samples.png
        ├── gradcam_000c1434d8d7.png
        └── preprocessing_comparison.png
```

---

# 📄 Source Code Description

| File                     | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `preprocess.py`          | Retinal cropping, resizing, padding and CLAHE |
| `cache_preprocessing.py` | Precomputes and caches processed images       |
| `split_dataset.py`       | Creates stratified training/validation splits |
| `dataset.py`             | PyTorch dataset and data loaders              |
| `model.py`               | EfficientNet-B0 baseline                      |
| `cbam.py`                | Channel and spatial attention implementation  |
| `cbam_model.py`          | EfficientNet-B0 + CBAM architecture           |
| `train_baseline.py`      | Baseline training and evaluation              |
| `train_cbam.py`          | CBAM training and evaluation                  |
| `predict.py`             | Single-image prediction                       |
| `gradcam.py`             | Grad-CAM generation                           |
| `dashboard.py`           | Streamlit application                         |
| `visualize_dataset.py`   | Dataset visualization                         |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Kirti-Official/cvciaproject.git
cd cvciaproject
```

Switch to the development branch if required:

```bash
git checkout abhinav-dr-project
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

Install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install opencv-python
pip install numpy pandas
pip install scikit-learn
pip install matplotlib
pip install pillow
pip install streamlit
```

---

# 📂 Dataset Setup

The dataset is intentionally **not included in the repository** because of its size and dataset distribution considerations.

Place the APTOS dataset locally in:

```text
dataset/
└── aptos/
    ├── train.csv
    └── train_images/
```

The CSV should contain:

```text
id_code
diagnosis
```

---

# ▶️ Running the Project

## Step 1 — Prepare the dataset

Create the train/validation split:

```bash
python src/split_dataset.py
```

## Step 2 — Preprocess images

Run the preprocessing cache:

```bash
python src/cache_preprocessing.py
```

## Step 3 — Train baseline

```bash
python src/train_baseline.py
```

## Step 4 — Train CBAM model

```bash
python src/train_cbam.py
```

The best CBAM checkpoint is expected at:

```text
models/efficientnet_b0_cbam_best.pth
```

Model checkpoints are excluded from Git tracking by `.gitignore`.

---

# 🔮 Single Image Prediction

After training, use:

```bash
python src/predict.py
```

The prediction pipeline:

```text
Image
 ↓
Preprocessing
 ↓
EfficientNet-B0 + CBAM
 ↓
Softmax Probabilities
 ↓
Predicted DR Class
```

---

# 🔥 Grad-CAM

To generate an explanation:

```bash
python src/gradcam.py
```

The resulting visualization is saved under:

```text
outputs/plots/
```

---

# 🌐 Streamlit Dashboard

Start the dashboard using:

```bash
python -m streamlit run src/dashboard.py
```

The Streamlit interface provides:

* image upload
* preprocessing visualization
* DR prediction
* probability distribution
* Grad-CAM heatmap
* Grad-CAM overlay
* model performance information

---

# 🧪 Reproducibility

The project uses a fixed random seed of:

```text
42
```

where applicable.

However, exact results can vary depending on:

* PyTorch version
* torchvision/timm version
* CPU implementation
* random initialization
* operating system
* hardware environment

The current reported results correspond to the experimental configuration used during development.

---

# ⚠️ Limitations

The current implementation has several limitations:

1. **Class imbalance**
   Minority classes, particularly severe DR, have fewer training examples.

2. **Neighboring-class confusion**
   Mild/moderate and severe/proliferative classes can be difficult to distinguish.

3. **Limited training schedule**
   The current experiment uses five training epochs because of computational constraints.

4. **No independent external validation**
   The current reported results are based on the APTOS train/validation experiment.

5. **Grad-CAM is not lesion detection**
   It provides model explanations rather than explicit lesion localization.

6. **Research prototype**
   The system is not intended to provide clinical diagnosis or replace professional ophthalmic assessment.

---

# 🔮 Future Work

Future development can extend the current pipeline in several directions.

### 1. Longer training

Use:

* more epochs
* learning-rate scheduling
* early stopping
* stronger regularization

### 2. Advanced imbalance handling

Investigate:

* Focal Loss
* class-balanced loss
* oversampling
* balanced batch sampling

### 3. External validation

Evaluate the model on additional retinal datasets such as:

* DDR
* IDRiD
* Messidor

### 4. Lesion detection and segmentation

Add an explicit lesion detection/segmentation component for structures such as:

* microaneurysms
* hemorrhages
* exudates

### 5. Transformer-based feature extraction

Future experiments can investigate Vision Transformer-based representations and CNN-transformer fusion.

### 6. Multi-task learning

The system could be extended to jointly predict:

```text
Image Quality
      +
DR Severity
      +
Lesion Information
```

### 7. Improved explainability

Future work can combine Grad-CAM with attention visualization and lesion annotations for more clinically meaningful explanations.

---

# 🧾 Current Implementation vs Future Architecture

| Component                    | Status         |
| ---------------------------- | -------------- |
| APTOS 2019                   | ✅ Implemented  |
| Five-class DR classification | ✅ Implemented  |
| Retinal cropping             | ✅ Implemented  |
| CLAHE                        | ✅ Implemented  |
| Data augmentation            | ✅ Implemented  |
| EfficientNet-B0              | ✅ Implemented  |
| Transfer learning            | ✅ Implemented  |
| Class-weighted loss          | ✅ Implemented  |
| CBAM                         | ✅ Implemented  |
| Baseline comparison          | ✅ Implemented  |
| Grad-CAM                     | ✅ Implemented  |
| Streamlit dashboard          | ✅ Implemented  |
| YOLOv8 lesion detection      | 🔮 Future work |
| Lesion segmentation          | 🔮 Future work |
| Vision Transformer branch    | 🔮 Future work |
| Cross-attention fusion       | 🔮 Future work |
| Multi-task learning          | 🔮 Future work |
| External dataset validation  | 🔮 Future work |

---

# 👥 Project Information

**Project:** Deep Learning for Diabetic Retinopathy Screening from Retinal Fundus Images

**Student:** Abhinav Singh
**Register Number:** 2362004

**Program:** B.Tech CSE (AIML)
**Institution:** CHRIST (Deemed to be University), Bengaluru

---

# 📚 References

1. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *International Conference on Machine Learning (ICML)*, 2019.

2. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," *European Conference on Computer Vision (ECCV)*, 2018.

3. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs," *JAMA*, 2016.

4. Ting et al., "Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From Multiethnic Populations With Diabetes," *JAMA*, 2017.

---

# ⚕️ Disclaimer

This project is developed for **academic and research purposes**.

The model predictions should not be considered a medical diagnosis. The system has not been clinically validated and should not be used as a replacement for examination or diagnosis by a qualified healthcare professional.

---

## ⭐ Acknowledgement

This project explores the application of computer vision, transfer learning and attention mechanisms to automated diabetic retinopathy screening from retinal fundus images.
