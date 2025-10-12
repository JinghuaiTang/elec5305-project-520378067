# Urban Sound Classification using Convolutional Neural Networks

This project aims to classify environmental sounds (such as sirens, car horns, dog barks, and drilling) using a Convolutional Neural Network (CNN). The main objective is to develop a model that can automatically recognize different types of urban sounds from the **UrbanSound8K** dataset and evaluate its accuracy and robustness under various noise conditions.  
This project also serves as a practical exercise in combining signal processing and deep learning methods, focusing on applying neural networks to real-world acoustic scenarios.

---

## Project Objectives

- Build a CNN-based model to classify urban environmental sounds.  
- Utilize publicly available datasets to ensure reproducibility.  
- Explore preprocessing techniques and data augmentation for audio signals.  
- Evaluate the model’s performance using established machine learning metrics.  
- Gain hands-on experience with Python, PyTorch, and audio feature extraction tools.

---

## Methodology

The workflow of this project consists of four major stages:

### 1. Data Preparation
- Use the **UrbanSound8K** dataset as the main data source, which includes 8,732 labeled sound clips across 10 urban sound categories.  
- Perform data preprocessing:  
  - Convert raw audio into **spectrograms** or **Mel-Frequency Cepstral Coefficients (MFCCs)**.  
  - Apply normalization and noise reduction to enhance feature quality.  
  - Use data augmentation techniques (e.g., adding random background noise, pitch shifting, or time stretching) to improve model generalization.

### 2. Model Design
- Implement a **Convolutional Neural Network (CNN)** for feature extraction and classification.  
- Experiment with different architectures to balance accuracy and computational cost.  
- Integrate traditional signal processing methods (such as filtering or spectral subtraction) as baseline comparisons.  
- Develop and test the model using **Python**, **PyTorch**, and **Librosa** for feature processing and visualization.

### 3. Training Process
- Train the CNN model using the **cross-entropy loss** function.  
- Apply optimizers such as **Adam** or **SGD** for parameter updates.  
- Use **mini-batch training** and monitor validation accuracy to prevent overfitting.  
- Tune hyperparameters (learning rate, batch size, number of epochs) for optimal convergence.

### 4. Performance Evaluation
- Evaluate model performance using:  
  - **Accuracy**, **Precision**, **Recall**, and **F1-score**.  
  - **Confusion Matrix** to visualize class-wise prediction results.  
- Test robustness under different noise levels to assess real-world applicability.

---

## Current Progress

- UrbanSound8K dataset has been downloaded and organized.  
- Preliminary preprocessing scripts (e.g., loading, feature extraction) have been tested.  
- Baseline CNN model implemented and initialized in PyTorch.  
- Initial runs on sample subsets show promising classification results (~60% accuracy after first epoch).  
- Ongoing work includes full dataset training and optimization.

---

## Expected Outcomes

- A functional CNN model capable of classifying common urban sounds with good accuracy.  
- A complete, reproducible pipeline from data preparation to evaluation.  
- Improved understanding of spectrogram-based feature extraction and CNN audio classification.  
- A reference project for applying AI in environmental sound recognition tasks.

---

## Repository Structure

- project_root/  
  - data/ — Dataset and preprocessing scripts  
  - model/ — CNN model architecture and training scripts  
  - results/ — Evaluation metrics and plots  
  - utils/ — Helper functions for data handling  
  - README.md — Project documentation

## Next Steps

- Complete training on the full UrbanSound8K dataset.  
- Visualize classification metrics and generate confusion matrices.  
- Compare CNN performance with traditional MFCC + SVM baseline.  
- Document experimental results and prepare final report.

---

## References

- UrbanSound8K Dataset: https://urbansounddataset.weebly.com/urbansound8k.html  
- PyTorch Documentation: https://pytorch.org/docs/  
- Librosa Audio Analysis Library: https://librosa.org/
