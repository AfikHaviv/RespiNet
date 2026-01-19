# RespiNet: Automatic Respiratory Pathology Detection 🫁

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**RespiNet** is a deep learning system designed to automate the diagnosis of respiratory diseases by analyzing lung sounds. By leveraging **Dilated Convolutional Neural Networks (CNNs)** and a rigorous data filtering pipeline, this model classifies breath cycles as **Healthy** or **Pathological** (containing crackles/wheezes) with **95% accuracy**, significantly outperforming traditional baseline models.

---

## 📌 Problem & Solution

### The Challenge
Respiratory diseases (COPD, Pneumonia, Bronchiectasis) are a leading cause of mortality. Traditional diagnosis relies on manual auscultation (listening with a stethoscope), which is:
* **Subjective:** Dependent on the doctor's hearing and experience.
* **Prone to Error:** Difficult in noisy environments.
* **Limited:** Standard CNNs often fail to capture long-duration auditory features (like continuous wheezing).

### Our Solution
1.  **Dilated CNN Architecture:** We utilize dilated convolutions to expand the "Receptive Field" of the network. This allows the model to capture long-term temporal dependencies (wheezes lasting >1s) without increasing computational cost.
2.  **Ambiguity Filtering:** A novel preprocessing step that removes "clean" breath cycles from sick patients, ensuring the model trains on distinct pathological features rather than noisy labels.
3.  **Balanced Training:** Implementation of a `WeightedRandomSampler` to handle the severe class imbalance (90% Sick vs. 10% Healthy) in the ICBHI dataset.

---

## 📊 Results & Performance

We evaluated the model on a strictly held-out validation set.

| Metric | Score | Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **94%** | Overall correctness on unseen data. |
| **Sensitivity (Recall)** | **98%** | The model almost never misses a sick patient (Crucial for screening). |
| **AUC Score** | **0.97** | Excellent capability to separate Healthy vs. Pathological classes. |

---

## 🛠️ Methodology

### 1. Dataset
We used the **ICBHI 2017 Respiratory Sound Database**:
* **920** Audio recordings (.wav).
* **126** Patients.
* **Annotations** for every breath cycle.

### 2. The Pipeline
1.  **Segmentation:** Raw audio is sliced into ~6,900 individual breath cycles using time-stamped annotations.
2.  **Feature Extraction:** Conversion of 1D audio waveforms into **Mel-Spectrograms** (visual heatmaps of sound frequencies).
3.  **Data Cleaning:**
    * *Input:* Mixed data with noisy labels.
    * *Process:* Discarding clean breaths labeled as "Sick".
    * *Output:* High-quality, distinct training samples.
4.  **Training:** The model is trained using **CrossEntropyLoss** and the **Adam** optimizer with a dynamic Learning Rate Scheduler.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for training)

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/RespiNet.git](https://github.com/AfikHaviv/RespiNet.git)
cd RespiNet
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Setup Data
1.  Download the **ICBHI 2017 Respiratory Sound Database**.
2.  Place the files in: `./data/Respiratory_Sound_Database/`
    * Should contain: `audio_and_txt_files/` and `patient_diagnosis.csv`.

### 4. Run the Notebook
1.  Open `RespiNet.ipynb` in Jupyter Lab or VS Code.
2.  **Run All Cells**: The notebook is structured to handle data loading, preprocessing, training, and evaluation sequentially.
3.  **Live Demo**: The final section of the notebook allows you to play a random audio file and see the model's prediction in real-time.

---

## 📂 Project Structure

```text
RespiNet/
├── data/
│   └── Respiratory_Sound_Database/
│       ├── audio_and_txt_files/     # Raw .wav and .txt files
│       └── patient_diagnosis.csv    # Diagnosis labels
│   └──processed_tensors/            # Cached .pt files (Generated during training)
├── RespiNet.ipynb                   # Main Jupyter Notebook
├── respin_net_model.pth             # Trained Model Weights
└── README.md                        # Project Documentation
```
## 👨‍💻 Authors

* **Afik Haviv**
* **Shir Molakandove**
* *Project for Deep Learning Course, 2026*

---

## 📜 Acknowledgments

* Dataset provided by the **ICBHI 2017 Challenge**.
* Built with **PyTorch** and **Librosa**.