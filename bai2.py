import numpy as np
from collections import Counter
import sys

class KNN:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
    
    def predict(self, X):
        predictions = []
        # Chuyển đổi sang numpy array để tính toán vector hóa
        X = np.array(X)
        
        for x in X:
            # Tính khoảng cách theo vector hóa
            if self.distance_metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:  # manhattan
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)

def load_data(filename):
    try:
        # Đọc file và xử lý từng dòng
        data = []
        with open(filename, 'r') as f:
            for line in f:
                # Thử các delimiter khác nhau
                if ',' in line:
                    values = line.strip().split(',')
                else:
                    values = line.strip().split()
                
                # Chuyển đổi sang float
                values = [float(val) for val in values]
                data.append(values)
        
        # Chuyển thành numpy array
        data = np.array(data)
        
        # Tách features và labels
        X = data[:, :-1]
        y = data[:, -1]
        
        # Chuyển đổi kiểu dữ liệu
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        return X, y
    
    except Exception as e:
        print(f"Error loading file {filename}: {str(e)}")
        sys.exit(1)

def calculate_confusion_matrix(y_true, y_pred):
    max_label = max(max(y_true), max(y_pred)) + 1
    n_classes = int(max_label)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix

def main():
    if len(sys.argv) != 4:
        print("Usage: python knn.py <trainset> <testset> <k>")
        sys.exit(1)
        
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    k = int(sys.argv[3])
    
    print(f"Loading training data from {train_file}...")
    X_train, y_train = load_data(train_file)
    
    print(f"Loading test data from {test_file}...")
    X_test, y_test = load_data(test_file)
    
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    
    print("Predicting...")  # Thêm thông báo để biết tiến độ
    y_pred = knn.predict(X_test)
    
    print("Calculating metrics...")
    conf_matrix = calculate_confusion_matrix(y_test, y_pred)
    accuracy = np.mean(y_pred == y_test)
    
    print("Confusion matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {accuracy}")

if __name__ == "__main__":
    main()