import json

with open("./results/actual.json", "r") as f:
    gt_data = json.load(f)

with open("./results/predictions.json", "r") as f:
    pred_data = json.load(f)

gt = {item["frame"]: item["label"] for item in gt_data}
pred = {item["frame"]: item["label"] for item in pred_data}

TP = TN = FP = FN = 0

for frame in gt:
    if frame not in pred:
        continue  

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

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
