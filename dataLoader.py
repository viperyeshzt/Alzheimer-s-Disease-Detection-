from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Paths
train_path = "Dataset\AugmentedAlzheimerDataset"
val_path = "Dataset\OriginalDataset"

# Training generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Validation generator
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Training data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=True
)

# Validation data
val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

print("Training Images:", train_data.samples)
print("Validation Images:", val_data.samples)
print("Classes:", train_data.class_indices)