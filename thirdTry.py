import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import scikitplot as skplt
import matplotlib.pyplot as plt

train_path = './SPECTF_train.csv'
test_path = './SPECTF_test.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)


X_train = train_data.iloc[0:80, 1:] 
y_train = train_data.iloc[0:80, 0]

X_test = test_data.iloc[0:187, 1:]
y_test = test_data.iloc[0:187, 0]

print(f'{X_train}\n{y_train}\n{X_test}\n{y_test}')

# Normalize the feature data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled[:5], X_test_scaled[:5]


# Define kernel and parameter ranges
kernels = ["linear", "rbf", "poly"]
C_values = [0.1, 1, 10, 100, 1000]
epsilon_values = [0.001, 0.01, 0.1, 1, 10]


results = []


for kernel in kernels:
    for C in C_values:
        start_time = time.time()
        
        
        model = SVC(kernel=kernel, C=C, degree=2 if kernel == "poly" else 3, probability=True)  
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        class_prob = model.predict_proba(X_test_scaled)        

        accuracy = accuracy_score(y_test, y_pred)
        runtime = time.time() - start_time
        results.append((kernel, C, accuracy, runtime))

        skplt.metrics.plot_roc(y_test, class_prob)
        plt.title(f'SVM, {kernel}, C - {C}')
        plt.show()

results_df = pd.DataFrame(results, columns=["Kernel", "C", "Accuracy", "Runtime (s)"])
results_df_sorted = results_df.sort_values(by=["Kernel", "C"], ascending=True).reset_index(drop=True)

# Print the results DataFrame
print("SVM Results with Different Kernels and Slack Variable (C):")
print(results_df_sorted)