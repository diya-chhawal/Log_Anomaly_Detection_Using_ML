# Log Anomaly Detection using Machine Learning and LightAD

## 📌 Overview
This project explores **log-based anomaly detection** on the HDFS dataset using both:
- Manual implementations of classical machine learning models
- The LightAD framework for systematic evaluation

The objective is to analyze whether **simple machine learning models** can effectively detect anomalies and compare their performance with more complex approaches.

---

## 🚀 Models Implemented

### 🔹 Manual Implementation
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)
- Multi-Layer Perceptron (MLP / SLFN)

### 🔹 LightAD Framework
- KNN
- Decision Tree
- SLFN (Single Layer Feedforward Network)

---

## 📊 Dataset

- **HDFS Log Dataset**
- Preprocessed into **5 shuffled splits** (`shuffle_0` → `shuffle_4`)
- Each split contains:
  - `x_train`, `x_test`
  - `y_train`, `y_test`

---

## 🧪 Methodology

### Step 1: Manual Model Implementation
- Loaded `.npz` dataset splits
- Trained models on training data
- Evaluated using **F1-score (anomaly class)** across 5 splits
- Averaged results

---

### Step 2: LightAD Framework Execution
- Used `main_hdfs.py` to run experiments
- Evaluated:
  - Precision
  - Recall
  - F1-score
  - Specificity
  - Balanced Accuracy
  - Training time
  - Inference time

---

## 📈 Results

### 🔹 Manual Results

| Model | Avg F1 (Anomaly) |
|------|------------------|
| KNN | 0.621 |
| Decision Tree | 0.615 |
| MLP | 0.615 |

---

### 🔹 LightAD Results

| Model | F1 | Train Time | Inference Time |
|------|----|-----------|----------------|
| KNN | 0.612 | Very Fast | Fast |
| Decision Tree | 0.615 (Best) | Fast | Very Fast |
| SLFN (MLP) | 0.611 | Slow | Fast |

---

## 📊 Visualization

![Model Comparison](results/model_comparison.png)

---

## 🧠 Key Observations

- All models achieve **similar F1-scores (~0.61–0.62)**
- **Decision Tree performs slightly better**, but the difference is minimal
- **KNN is fastest**, while MLP is significantly slower
- Dataset contains **highly repetitive log patterns**
- **Low recall (~0.45)** indicates difficulty in detecting rare anomalies

---

## 🧠 Key Insights

- Log anomaly detection is a **binary classification problem**
- Simple models perform **as well as complex models**
- Deep learning is **not necessary** for this dataset
- Efficiency (time) is a critical factor

---

## 🧾 Conclusion

This project demonstrates that **simple machine learning models such as KNN and Decision Tree achieve comparable performance** in log anomaly detection tasks.

Despite increased complexity, neural networks do not provide significant improvements. These findings align with research showing that **simple models are often sufficient for structured and repetitive log data**.

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
