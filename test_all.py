import os
import subprocess

def run_knn(train_file, test_file, k):
    cmd = f"python bai2.py {train_file} {test_file} {k}"
    subprocess.run(cmd, shell=True)

# Định nghĩa các dataset và đường dẫn
datasets = {
    'iris': ('data/iris/iris.trn', 'data/iris/iris.tst'),
    'faces': ('data/faces/data.trn', 'data/faces/data.tst'),
    'fp': ('data/fp/fp.trn', 'data/fp/fp.tst'),
    'letter': ('data/letter/let.trn', 'data/letter/let.tst'),
    'leukemia': ('data/leukemia/ALLAML.trn', 'data/leukemia/ALLAML.tst'),
    'optics': ('data/optics/opt.trn', 'data/optics/opt.tst')
}

# Test chỉ với k = 1 và k = 3
k_values = [1, 3]

for dataset_name, (train_file, test_file) in datasets.items():
    print(f"\n=== Testing {dataset_name} dataset ===")
    
    # Kiểm tra file training
    if not os.path.exists(train_file):
        print(f"Warning: Training file {train_file} not found. Skipping dataset.")
        continue
        
    # Kiểm tra file test
    if not os.path.exists(test_file):
        print(f"Warning: Test file {test_file} not found. Skipping dataset.")
        continue
        
    for k in k_values:
        print(f"\nk = {k}")
        try:
            run_knn(train_file, test_file, k)
        except Exception as e:
            print(f"Error running KNN on {dataset_name} with k={k}: {str(e)}") 