from tensorflow.keras.models import load_model
from dataLoader import val_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained model
model = load_model("best_alzheimer_model.keras")

# Evaluate accuracy
loss, accuracy = model.evaluate(val_data)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy * 100, "%")

# Predictions
predictions = model.predict(val_data)
pred_classes = np.argmax(predictions, axis=1)

true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, pred_classes)

print("\nConfusion Matrix:\n")
print(cm)