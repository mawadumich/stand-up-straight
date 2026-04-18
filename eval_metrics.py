import json

# Load files
with open("ground_truth.json", "r") as f:
    gt_data = json.load(f)

with open("predictions.json", "r") as f:
    pred_data = json.load(f)

# Convert to dicts keyed by frame for fast lookup
gt = {item["frame"]: item["label"] for item in gt_data}
pred = {item["frame"]: item["label"] for item in pred_data}

# Initialize counts
TP = TN = FP = FN = 0

# Compare
for frame in gt:
    if frame not in pred:
        continue  # or raise error if strict matching is required

    y_true = gt[frame]
    y_pred = pred[frame]

    if y_true == 1 and y_pred == 1:
        TP += 1
    elif y_true == 0 and y_pred == 0:
        TN += 1
    elif y_true == 0 and y_pred == 1:
        FP += 1
    elif y_true == 1 and y_pred == 0:
        FN += 1

# Print results
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

# Optional: accuracy, precision, recall, F1
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
