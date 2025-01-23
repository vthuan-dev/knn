import numpy as np

# Training data
X_train = np.array([
    [0.376000, 0.488000],
    [0.312000, 0.544000],
    [0.298000, 0.624000],
    [0.394000, 0.600000],
    [0.506000, 0.512000],
    [0.488000, 0.334000],
    [0.478000, 0.398000],
    [0.606000, 0.366000],
    [0.428000, 0.294000],
    [0.542000, 0.252000]
])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Test data
X_test = np.array([
    [0.550000, 0.364000],
    [0.558000, 0.470000],
    [0.456000, 0.450000],
    [0.450000, 0.570000]
])

#áp dụng thuật toán euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

#áp dụng thuật toán manhattan 
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# để dự đoán kết quả
def knn_predict(X_train, y_train, x_test, k, metric='euclidean'):
    distances = []
    for x_train in X_train:
        if metric == 'euclidean':
            dist = euclidean_distance(x_train, x_test)
        else:
            dist = manhattan_distance(x_train, x_test)
        distances.append(dist)
    
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]
    return np.bincount(k_nearest_labels).argmax()

def print_results(k, metric):
    print(f"\nWith k = {k}")
    print(f"Using {metric} distance:")
    print("+--------+--------+--------+")
    print("| X1     | X2     | Class  |")
    print("+========+========+========+")
    
    for x_test in X_test:
        y_pred = knn_predict(X_train, y_train, x_test, k, metric)
        print(f"| {x_test[0]:.3f} | {x_test[1]:.3f} | {y_pred:^6} |")
        print("+--------+--------+--------+")

#kết quả với k=1
print_results(k=1, metric='euclidean')
print_results(k=1, metric='manhattan')

# In kết quả với k=3
print_results(k=3, metric='euclidean')
print_results(k=3, metric='manhattan')
