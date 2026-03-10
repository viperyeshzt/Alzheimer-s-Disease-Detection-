# -----------------------------------
# Import Libraries
# -----------------------------------
from dataLoader import train_data, val_data
from model import model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------------------
# Dataset Information
# -----------------------------------
print("Number of Training Images:", train_data.samples)
print("Number of Validation Images:", val_data.samples)
print("Class Labels:", train_data.class_indices)

# -----------------------------------
# Compute Class Weights (for imbalance)
# -----------------------------------
labels = train_data.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("Computed Class Weights:", class_weights)

# -----------------------------------
# Callbacks
# -----------------------------------

# Stop training if validation stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Save the best model automatically
checkpoint = ModelCheckpoint(
    "best_alzheimer_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Reduce learning rate if validation loss stops improving
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# -----------------------------------
# Train Model
# -----------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    steps_per_epoch=800,
    validation_steps=200,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# -----------------------------------
# Save Final Model
# -----------------------------------
model.save("final_alzheimer_model.keras")

print("Training completed successfully.")
print("Final model saved as: final_alzheimer_model.keras")