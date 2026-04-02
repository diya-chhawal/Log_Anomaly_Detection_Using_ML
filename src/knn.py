import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

f1_scores = []

for i in range(5):
    data = np.load(f"datasets/splited_datasets/hdfs/shuffle_{i}.npz")

    X_train = data["x_train"]
    X_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, pos_label="+")
    f1_scores.append(f1)

    print(f"Shuffle {i} F1:", f1)

print("\nAverage F1:", sum(f1_scores)/len(f1_scores))