# =========================================================
# DECODE LABS: PROJECT 2 - DATA CLASSIFICATION USING AI
# =========================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

print("Initializing Project 2: Supervised Learning Pipeline (KNN Alignment)...\n")

# ---------------------------------------------------------
# Step 1: Load the Dataset (Iris Benchmark)
# ---------------------------------------------------------
# 150 Balanced Samples, 3 Classes, 4 Dimensions
iris = load_iris()
X = iris.data  
y = iris.target  

# ---------------------------------------------------------
# Step 2: Preprocessing (The Gatekeeper Rule: Scaling)
# ---------------------------------------------------------
# Apply StandardScaler (Mean = 0, Variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# Step 3: Structural Integrity (The Split)
# ---------------------------------------------------------
# Shuffle and split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, shuffle=True
)

# ---------------------------------------------------------
# Step 4: Apply Simple Classification Algorithm (KNN)
# ---------------------------------------------------------
# Blueprint: Initialize K-Nearest Neighbors 
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 5: Evaluation & Verification (Metrics)
# ---------------------------------------------------------
y_pred = knn_model.predict(X_test)

# Calculating the required validation targets from the IPO output layer
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("=============================================")
print("             VERIFICATION REPORT             ")
print("=============================================")
print(f" Model Selected : K-Nearest Neighbors (KNN)")
print(f" Test Accuracy  : {accuracy * 100:.2f}%")
print(f" Macro F1 Score : {macro_f1:.4f}")
print("---------------------------------------------")
print(" CONFUSION MATRIX:")
print(conf_matrix)
print("\n Matrix Layout:")
print(f"  Rows   : Actual Classes ({', '.join(iris.target_names)})")
print(f"  Columns: Predicted Classes ({', '.join(iris.target_names)})")
print("=============================================")