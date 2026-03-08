from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------------
# Dataset Path
# -----------------------------------
dataset_path = "C:/Users/HP/OneDrive/Desktop/Alzheimer/Datasets/archive (1)/OriginalDataset"

# -----------------------------------
# Training Data Generator
# -----------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

# -----------------------------------
# Validation Data Generator
# -----------------------------------
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# -----------------------------------
# Load Training Data
# -----------------------------------
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# -----------------------------------
# Load Validation Data
# -----------------------------------
val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -----------------------------------
# Dataset Information
# -----------------------------------
print("\nDataset Loaded Successfully")
print("Number of Training Images:", train_data.samples)
print("Number of Validation Images:", val_data.samples)
print("Class Labels:", train_data.class_indices)