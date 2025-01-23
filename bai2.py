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
        print(f"\nPredicting using {self.distance_metric} distance...")
        total = len(X)
        
        # Xử lý theo batch để tăng tốc
        batch_size = 100
        for i in range(0, len(X), batch_size):
            batch = X[i:min(i + batch_size, len(X))]
            batch_predictions = []
            
            print(f"Processing {i}/{total} samples...", end='\r')
            
            for x in batch:
                if self.distance_metric == 'euclidean':
                    distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
                else:  # manhattan
                    distances = np.sum(np.abs(self.X_train - x), axis=1)
                    
                k_indices = np.argpartition(distances, self.k)[:self.k]
                k_nearest_labels = self.y_train[k_indices]
                most_common = Counter(k_nearest_labels).most_common(1)
                batch_predictions.append(most_common[0][0])
                
            predictions.extend(batch_predictions)
            
        print(f"\nCompleted processing {total} samples!")
        return np.array(predictions)

def load_data(filename):
    try:
        try:
            data = np.loadtxt(filename, delimiter=',')
        except:
            data = np.loadtxt(filename, delimiter=' ')
            
        X = data[:, :-1]
        y = data[:, -1]
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
    
    print(f"\nLoading training data from {train_file}...")
    X_train, y_train = load_data(train_file)
    print(f"Training data shape: {X_train.shape}")
    
    print(f"Loading test data from {test_file}...")
    X_test, y_test = load_data(test_file)
    print(f"Test data shape: {X_test.shape}")
    
    # Test với cả 2 loại khoảng cách
    for distance in ['euclidean', 'manhattan']:
        print(f"\nTesting with {distance} distance:")
        knn = KNN(k=k, distance_metric=distance)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracy = np.mean(y_pred == y_test)
        conf_matrix = calculate_confusion_matrix(y_test, y_pred)
        
        print(f"\nResults for k={k} using {distance} distance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)

if __name__ == "__main__":
    main()