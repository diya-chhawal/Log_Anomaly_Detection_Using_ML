import matplotlib.pyplot as plt

# Model names
models = ["KNN", "Decision Tree", "MLP"]

# F1 scores (from your results)
f1_scores = [0.612, 0.615, 0.611]

plt.figure()
plt.bar(models, f1_scores)

plt.title("Model Comparison (F1 Score - Anomaly Detection)")
plt.xlabel("Models")
plt.ylabel("F1 Score")

# Show graph
plt.show()