# Evaluating Classical Spectral Features and Shallow Classifiers for Urban Sound Classification

This project investigates how far a carefully designed combination of **classical spectral features** and **shallow classifiers** can go for **urban sound classification**, using the **UrbanSound8K** dataset. Instead of relying on deep neural networks, we focus on compact, interpretable feature representations (e.g. MFCC and simple spectral descriptors) together with lightweight models such as **logistic regression**, **RBF-kernel SVM**, and **random forest**.

The work is part of the ELEC5305 project and is designed to be **reproducible, computationally light, and easy to analyse**, making it suitable both as a student project and as a baseline for future, more complex systems.

---

## Research Question

> **Given MFCC-based and related classical spectral features extracted from each audio clip, how well can lightweight shallow models (logistic regression, RBF SVM, random forest) distinguish between the ten UrbanSound8K classes—especially in terms of macro-averaged F1 under realistic computational constraints—when compared with a trivial majority-class baseline?**

We are particularly interested in **balanced performance across all classes**, not just high overall accuracy, so **macro-F1** is a key metric in this project.

---

## Project Objectives

- Build a **complete classical feature pipeline** for UrbanSound8K:
  - Parse metadata and audio files across all 10 folds.
  - Extract fixed-length feature vectors based on **MFCC statistics** and **global spectral descriptors** (spectral centroid, bandwidth, roll-off, flatness, RMS energy, zero-crossing rate, and deltas).
- Train and compare several **shallow classifiers**:
  - Multinomial **logistic regression**  
  - **Support Vector Machine (SVM)** with RBF kernel  
  - **Random Forest**
- Evaluate performance using:
  - **Accuracy** and **macro-averaged F1-score**
  - Per-class **precision**, **recall**, and **F1**
  - **Confusion matrices** and **baseline comparison** against a majority-class predictor
- Analyse:
  - **Per-class behaviour** and common confusions
  - **Feature importance** rankings (from the random forest)
  - **Class distribution** and dataset statistics
- Keep the full pipeline:
  - **Lightweight** (runs comfortably on a laptop)
  - **Transparent** (no black-box deep models)
  - **Reproducible** (single script can re-generate all CSVs and figures)

---

## Methodology Overview

The implementation is organised as a **modular pipeline**, with each stage responsible for one part of the experiment:

### 1. Data and Feature Engineering

- Load and subset **UrbanSound8K** metadata (`UrbanSound8K.csv`) by folds and classes.
- Map class names (e.g. `air_conditioner`, `car_horn`, `dog_bark`, …) to integer labels.
- For each audio file:
  - Load waveform at a fixed sampling rate and duration.
  - Compute **MFCCs** (with optional deltas and delta-deltas).
  - Compute classical spectral descriptors:
    - Spectral centroid, bandwidth, roll-off, flatness  
    - RMS energy, zero-crossing rate
  - Aggregate frame-level features into global statistics (mean, standard deviation).
- Assemble a dense feature matrix **X ∈ ℝ^(N × D)** (about 8686×90) and label vector **y**.
- Support **feature caching** to avoid recomputing MFCCs and spectra across runs.

### 2. Experimental Protocol

- Perform **stratified train/validation/test split**:
  - Preserve class distribution across splits.
  - Keep a **held-out test set** untouched for final evaluation.
- Compute a **majority-class baseline**:
  - Always predicts the most frequent class.
  - Provides lower-bound accuracy and macro-F1 for comparison.
- Optionally run **stratified K-fold cross-validation** on the full dataset to obtain:
  - Mean / std accuracy
  - Mean / std macro-F1  
  for each shallow model.

### 3. Shallow Classifiers and Training

- All classifiers are implemented as **scikit-learn pipelines**:
  - For SVM and logistic regression: `StandardScaler` → classifier  
  - For random forest: classifier directly on raw features.
- Train models on the **train + validation** data and evaluate on:
  - Validation split (for sanity checks)
  - Held-out test split (for final reported results)
- Metrics recorded:
  - Accuracy and macro-F1 on validation and test
  - Full **classification report** on the test split

### 4. Diagnostics, Analysis and Visualisation

- **Dataset diagnostics**:
  - Class distribution (counts and percentages)
  - Bar plots of class balance
- **Per-class metrics**:
  - Precision, recall and F1-score per class for each classifier
  - Stored as CSV tables for external analysis
- **Confusion matrices**:
  - Heat maps per classifier
  - Reveal which classes are most frequently confused
- **Random forest feature importance**:
  - CSV ranking of features by importance
  - Highlights which MFCC and spectral statistics matter most
- **Example spectrograms per class**:
  - Log-mel spectrograms for selected clips  
  - Provide intuitive connection between raw audio and statistical features

---

## Current Status

- Complete **feature-extraction pipeline** for UrbanSound8K implemented.
- All 10 classes and 10 folds are processed into a ~90-dimensional feature space.
- Shallow classifiers (logistic regression, RBF SVM, random forest) are trained and evaluated using:
  - 10-fold cross-validation
  - A fixed stratified train/validation/test split
- Key results (approximate):
  - **SVM (RBF)**:  
    - Test accuracy ≈ 0.89  
    - Test macro-F1 ≈ 0.89
  - **Random Forest**:  
    - Test accuracy ≈ 0.87  
    - Test macro-F1 ≈ 0.87
  - **Logistic Regression**:  
    - Test accuracy ≈ 0.75  
    - Test macro-F1 ≈ 0.76
  - **Majority-class baseline**:  
    - Accuracy ≈ 0.11  
    - Macro-F1 ≈ 0.02  
- Extensive CSV logs and figures have been generated for:
  - Cross-validation summary  
  - Train/val/test split statistics  
  - Per-class metrics per classifier  
  - Random forest feature importance  
  - Class distribution and confusion matrices

These results show that **classical spectral features + shallow classifiers** can reach close to **0.89 macro-F1** on UrbanSound8K, while staying **computationally lightweight** and **interpretable**.

---

## Repository Structure

A possible structure for this repository is:

- `Urban_sounds_classification.py` – main experiment script that runs the full pipeline  
- `Dataset/`  
  - `audio/` – UrbanSound8K audio files organised as `fold1` … `fold10`  
  - `metadata/` – `UrbanSound8K.csv` and related files  
- `results/`  
  - `train_val_test_summary.csv`  
  - `cross_validation_summary.csv`  
  - `baseline_majority_class_summary.csv`  
  - `per_class_metrics_*.csv` (for each classifier)  
  - `random_forest_feature_importances.csv`
- `figures/`  
  - Class distribution plots  
  - Confusion matrices  
  - Spectrogram examples per class
- `README.md` – project documentation (this file)

(Names and paths may be slightly different depending on the local setup.)

---

## What Problem Are We Trying to Solve?

In many recent works, deep CNN or Transformer models are used for urban sound classification, while **classical spectral features + shallow models** are often treated only as quick baselines with limited analysis. This project tries to answer:

- *“How strong can a well-engineered classical pipeline actually be?”*  
- *“Which classical spectral features are most informative for UrbanSound8K?”*  
- *“Can we achieve high, balanced macro-F1 with shallow models under realistic resource constraints?”*

By fully documenting the pipeline, metrics, and analysis around **MFCC-based spectral features** and **shallow classifiers**, this repository aims to provide a **transparent, strong baseline** that later deep-learning approaches can be fairly compared against.

---

## Future Extensions

Potential extensions include:

- Exploring **feature selection** to reduce dimensionality while preserving performance.
- Adding more **classical features** (e.g. chroma, spectral contrast, modulation features).
- Evaluating **additional shallow models** (e.g. gradient boosting, calibrated ensembles).
- Stress-testing **robustness** under synthetic noise, reverberation, and device response shifts.
- Performing **cross-dataset evaluation** with other urban/environmental sound corpora.
- Packaging the pipeline as a small **toolkit** for lightweight urban sound experiments.

---

## References

- UrbanSound8K Dataset: https://urbansounddataset.weebly.com/urbansound8k.html  
- scikit-learn: https://scikit-learn.org/  
- Librosa Audio Analysis Library: https://librosa.org/
