# K-Nearest Neighbors (KNN) Species Classification

A clean and end-to-end implementation of the **K-Nearest Neighbors (KNN)** algorithm using `scikit-learn` for multi-class classification on the classic **Iris Dataset**.

---

## 📌 Theoretical Overview

**K-Nearest Neighbors (KNN)** is a non-parametric, instance-based **lazy learning algorithm**. Instead of constructing a generalized model during training, it stores (memorizes) the training dataset and delays computation until evaluation/inference time.

### Key Algorithmic Steps:
1. **Memorization:** Store training feature vectors $\mathbf{X}$ and corresponding labels $y$.
2. **Distance Calculation:** Compute the Euclidean distance between target query points and all stored training samples:
   $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
3. **Neighbor Selection:** Identify the top $K$ nearest data points based on the computed distances.
4. **Majority Voting:** Determine predicted class assignments via majority voting among the $K$-nearest neighbors.

> **Best Practice:** Always select an **odd value for $K$** (e.g., $K=3$) in binary or balanced multi-class scenarios to prevent tie votes during decision boundary calculation.

---

## 🛠️ Pipeline Architecture

1. **Data Ingestion:** Load 4 numeric feature attributes from the Iris dataset:
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)
2. **Dataset Partitioning:** Execute an $80/20$ stratified train-test split (`test_size=0.2`).
3. **Model Configuration:** Initialize `KNeighborsClassifier` with hyperparameter choice $K = 3$.
4. **Fitting & Evaluation:** Train the algorithm and evaluate classification performance via standard accuracy evaluation.

---

## 📊 Performance & Evaluation

The classification pipeline achieves high accuracy evaluation metrics across all target classes (*setosa*, *versicolor*, *virginica*):

- **Selected Hyperparameter:** $K = 3$
- **Test Set Accuracy:** `96.67%`

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn
