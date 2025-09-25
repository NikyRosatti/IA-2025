
TOTAL = 200

TP = 30
TN = 140
FP = 20
FN = 10

accuracy = (TP + TN) / TOTAL
precision = TP / (TP + FP)
recall = TP / (TP + FN)

F1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")   
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {F1_score:.2f}")