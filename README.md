# Predictive Maintenance for 3-Phase Induction Motors using DANN

## 📌 Project Overview
This repository contains a machine learning pipeline designed to monitor the health of three-phase induction motors. By leveraging a data-driven approach, this system provides robust, real-time diagnostics that adapt to varying operational conditions, completely bypassing the limitations and scaling issues of traditional physics-based models. 

The core of this architecture is a **Domain-Adversarial Neural Network (DANN)**. While standard neural networks often fail across different hardware due to "Domain Shift" (memorizing a specific machine's unique noise profile), this DANN actively unlearns motor-specific traits to achieve true cross-machine generalization.

---

## ⚙️ Data Preprocessing & Pipeline
To process raw sensor data without relying on a "healthy" baseline reference:

* **Windowing and FFT:** Raw current readings are segmented into one-second intervals and converted to the frequency domain using a Fast Fourier Transform (FFT) to standardize frequency bins at 1 Hz.
* **Dynamic Normalization:** Each spectrum is dynamically scaled against the peak of its own fundamental frequency.
* **Tensor Construction:** The normalized data is capped at 1000 Hz, and the three electrical phases are stacked into a `(1000, 3)` tensor, allowing the network to evaluate inter-phase relationships.

---

## 🧠 Model Architecture
Because variable loads alter motor slip and shift fault frequencies, a 1D-Convolutional Neural Network (1D-CNN) serves as the primary feature extractor. The DANN utilizes two competing output branches:

1. **Feature Extractor (1D-CNN):** Captures broad harmonics and fine spectral details, compressing the data into a unified core feature set.
2. **Output Branch 1 (Fault Predictor):** Classifies the physical state of the motor (e.g., Healthy, Broken Rotor Bar, Inner Race, Outer Race) using a Softmax activation.
3. **Output Branch 2 (Adversarial Domain Predictor):** Attempts to identify which specific motor the data originated from. Connected via a Custom Gradient Reversal Layer (GRL), it mathematically punishes the feature extractor if it guesses correctly, forcing the model to focus purely on universal fault signatures.

---

## 🛠️ Hardware Deployment
The model's ability to generalize was validated through a live, real-world laboratory deployment. 

* **Hardware Setup:** An ESP32 microcontroller paired with three current sensors captures live Phase A, B, and C currents.
* **Live Inference:** Live data is routed through the exact same dynamic normalization pipeline and evaluated by the DANN. In physical testing, the model successfully translated its open-source training to custom hardware.

---

## 📊 Datasets & References
To force the adversarial network to learn universal signatures, it was trained on highly diverse data sourced from multiple open-source experiments. 

* [1] R. A. Das, "Motor Fault Dataset," Kaggle, 2024. Available: https://www.kaggle.com/code/rickaryadas/motor-fault-classification/input
* [2] W. Jung et al., "Vibration and Motor Current Dataset of Rolling Element Bearing Under Varying Speed Conditions for Fault Diagnosis: Subset 1," Mendeley Data, V7, 2023, doi:[ 10.17632/vxkj334rzv.7.](https://data.mendeley.com/datasets/vxkj334rzv/7)
* [3] W. Jung et al., "Vibration and Motor Current Dataset of Rolling Element Bearing Under Varying Speed Conditions for Fault Diagnosis: Part 2," Mendeley Data, V4, 2022, doi: [10.17632/x3vhp8t6hg.4.](https://data.mendeley.com/datasets/x3vhp8t6hg/4)
* [4] D. Kumar et al., "Current Signature Dataset of Three-Phase Induction Motor under Varying Load Conditions," Mendeley Data, V1, 2022, doi: [10.17632/gxdd74czwh.1.](https://data.mendeley.com/datasets/gxdd74czwh/1)
