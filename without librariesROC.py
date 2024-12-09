# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:44:41 2024

@author: Carlo
"""

import pandas as pd
import numpy as np
import time
from numpy.linalg import norm
from cvxopt import matrix, solvers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

train_path = './SPECTF_train.csv'
test_path = './SPECTF_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

test_data.columns = train_data.columns

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * norm(x1 - x2) ** 2)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

# SVM Training using Quadratic Programming
def train_svm(X, y, kernel, C):
    n_samples, n_features = X.shape
    K = np.array([[kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
    
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))

    G_std = np.diag(-np.ones(n_samples))
    G_slack = np.eye(n_samples)
    G = matrix(np.vstack((G_std, G_slack)))

    h_std = np.zeros(n_samples)
    h_slack = np.ones(n_samples) * C
    h = matrix(np.hstack((h_std, h_slack)))

    A = matrix(y, (1, n_samples), 'd')
    b = matrix(0.0)

    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)

    alphas = np.ravel(solution['x'])
    support_vectors = alphas > 1e-5

    w = np.sum(alphas[support_vectors, None] * y[support_vectors, None] * X[support_vectors], axis=0)
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))

    return w, b

def decision_function(X, w, b):
    return np.dot(X, w) + b

kernels = {
    "linear": linear_kernel,
    "rbf": lambda x1, x2: rbf_kernel(x1, x2, gamma=0.1),
    "poly": lambda x1, x2: polynomial_kernel(x1, x2, degree=2)
}

C_values = [0.1, 1, 10, 100, 1000]
results = []

for C in C_values:
    plt.figure(figsize=(8, 6))  # Create a new plot for each C
    for kernel_name, kernel in kernels.items():
        start_time = time.time()
           
        w, b = train_svm(X_train_scaled, y_train * 2 - 1, kernel, C)  # Adjust labels to -1 and 1
        y_scores = decision_function(X_test_scaled, w, b)  # Decision scores for ROC
        y_pred = (y_scores > 0).astype(int)  

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        accuracy = np.mean(y_pred == y_test)
        runtime = time.time() - start_time
        results.append((kernel_name, C, accuracy, roc_auc, runtime))

        plt.plot(fpr, tpr, label=f'{kernel_name}, AUC={roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line for random classifier
    plt.title(f"ROC Curves for Slack Variable C={C}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

results_df = pd.DataFrame(results, columns=["Kernel", "C", "Accuracy", "AUC", "Runtime (s)"])
results_df_sorted = results_df.sort_values(by=["Kernel", "C"], ascending=True).reset_index(drop=True)

print("SVM Results with Different Kernels and Slack Variable (C):")
print(results_df_sorted)
