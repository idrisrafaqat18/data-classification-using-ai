# Data Classification Using AI

> **Module 02: Data Classification Using AI**  
> DecodeLabs Industrial Training Kit | Batch 2026

---

## 📌 Project Overview

This project serves as a critical bridge from **rule-based heuristic logic** to true **Supervised Learning**. Rather than explicitly writing complex nested rules, this system leverages historical vector features to enable machines to derive optimal decision boundaries independently.

Using the **Iris Benchmark Dataset**, this module implements a complete machine learning pipeline including:
- Raw data ingestion and exploration
- Feature scaling and validation
- Random shuffling and train-test splits
- Algorithmic pattern recognition
- Performance metrics and reporting

## ✨ Key Features

### 🎯 IPO Framework Implementation

#### **1. Input Layer**
- **Dataset Domain**: Iris Benchmark Dataset
  - 150 perfectly balanced samples
  - 3 classes: Setosa, Versicolor, Virginica
  - 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width

- **Feature Scaling Strategy**: StandardScaler normalization
  - Transforms raw input domains into balanced boundaries
  - Centers data around Mean = 0, Variance = 1
  - Eliminates scale bias for fair comparison

#### **2. Process Layer**
- **Data Splitting Strategy**:
  - Randomized shuffle to eliminate sequence/order bias
  - 80% Training Set (pattern recognition)
  - 20% Test Set (unseen validation)

- **Classification Algorithm**: K-Nearest Neighbors (KNN)
  - Lazy-learning metric clustering approach
  - Categorizes new samples based on vector proximity distances

#### **3. Output Layer**
- **Accuracy Score**: Measures true generalization capability on unseen data
- **Confusion Matrix**: Cross-tabulation showing true vs. predicted class distribution
- **F1 Score**: Macro-averaged performance metric for multi-class harmony

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
scikit-learn
numpy
pandas
matplotlib (for visualizations)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/idrisrafaqat18/data-classification-using-ai.git

# Navigate to project directory
cd data-classification-using-ai

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the classification pipeline
python main.py

# Or run with custom parameters
python main.py --test-size 0.2 --random-state 42
```

---

## 📊 Dataset Information

| Aspect | Details |
|--------|---------|
| **Dataset** | Iris Benchmark |
| **Samples** | 150 |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Features** | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| **Train/Test Split** | 80/20 |

---

## 🛠️ Technology Stack

| Component | Purpose |
|-----------|---------|
| **scikit-learn** | ML pipeline, KNN algorithm, metrics |
| **pandas** | Data manipulation and exploration |
| **numpy** | Numerical computations |
| **matplotlib/seaborn** | Data visualization |

---

## 📈 Performance Metrics

The model evaluation includes three key metrics:

1. **Accuracy**: Overall correctness percentage on test set
2. **Confusion Matrix**: Detailed breakdown of classification results
3. **F1 Score (Macro)**: Balanced performance metric accounting for class imbalance

### Expected Output Example
```
Accuracy Score: 0.97
Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
F1 Score (Macro): 0.97
```

---

## 🎓 Learning Outcomes

Students mastering this project will understand:

✅ Ingesting and evaluating supervised baseline vector records  
✅ Data normalization workflows and scale normalization matrices  
✅ Avoiding model bias via randomized data shuffling  
✅ Implementing lazy-learning metric clustering concepts (KNN)  
✅ Reading and interpreting confusion matrices and F1 scores  
✅ Building end-to-end ML pipelines with scikit-learn  

---

## 📁 Project Structure

```
data-classification-using-ai/
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── iris.csv
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── model.py
    └── evaluation.py
```

---

## 🔄 Pipeline Workflow

```
1. Load Dataset (Iris)
   ↓
2. Explore & Validate Data
   ↓
3. Apply Feature Scaling (StandardScaler)
   ↓
4. Randomize & Split Data (80% train, 20% test)
   ↓
5. Train KNN Model
   ↓
6. Generate Predictions
   ↓
7. Calculate Metrics (Accuracy, Confusion Matrix, F1 Score)
   ↓
8. Output Results
```

---

## 📝 Code Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is part of the DecodeLabs Industrial Training Kit (Batch 2026).

---

## 👨‍💼 Author

**Idris Rafaqat**  
[GitHub Profile](https://github.com/idrisrafaqat18)

---

## 📚 References & Resources

- [Iris Dataset Documentation](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [Confusion Matrix Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
- [F1 Score Explanation](https://scikit-learn.org/stable/modules/model_evaluation.html#f1-score)
- [StandardScaler Reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

## ⚡ Quick Tips

- **Tuning KNN**: Experiment with different `n_neighbors` values (3, 5, 7)
- **Random State**: Use `random_state=42` for reproducible results
- **Feature Importance**: Try visualizing feature distributions with matplotlib
- **Cross-Validation**: Consider k-fold cross-validation for more robust metrics

---

**Happy Learning! 🎯**
