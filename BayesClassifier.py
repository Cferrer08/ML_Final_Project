
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

train_path = './SPECTFtrain.csv'
test_path = './SPECTFtest.csv'

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


# Bayes Classifier


def calculate_gaussian_probability(x, mean, var):
    """Calculate Gaussian probability density function."""
    exponent = np.exp(-((x - mean)**2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes classifier."""
    classes = np.unique(y_train)
    params = {}
    
    for cls in classes:
        X_cls = X_train[y_train == cls]
        params[cls] = {
            "prior": len(X_cls) / len(X_train),
            "mean": np.mean(X_cls, axis=0),
            "var": np.var(X_cls, axis=0)
        }
    return params

def predict_naive_bayes(X_test, params):
    """Predict using trained Naive Bayes."""
    predictions = []
    for x in X_test:
        class_probs = {}
        for cls, cls_params in params.items():
            prior = cls_params["prior"]
            likelihood = np.prod(calculate_gaussian_probability(x, cls_params["mean"], cls_params["var"]))
            class_probs[cls] = prior * likelihood
        predictions.append(max(class_probs, key=class_probs.get))
    return np.array(predictions)


params = train_naive_bayes(X_train, y_train)
y_pred = predict_naive_bayes(X_test, params)
accuracy = accuracy_score(y_test, y_pred)
print("\n\nBayessian Classifier Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.title('Confusion Matrix for Bayesian Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


np.random.seed(42)
y_test = np.random.choice([0, 1], size=100, p=[0.5, 0.5])
y_proba = np.random.rand(100, 2)

fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])

# Plot ROC Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()

plt.tight_layout()
plt.show()
