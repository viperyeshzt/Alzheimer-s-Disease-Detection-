from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# ---------------------------------------
# Load Pretrained EfficientNetB0
# ---------------------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# ---------------------------------------
# Freeze early layers
# ---------------------------------------
for layer in base_model.layers[:-60]:
    layer.trainable = False

# Unfreeze deeper layers
for layer in base_model.layers[-60:]:
    layer.trainable = True

# ---------------------------------------
# Custom Classification Head
# ---------------------------------------
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

# Dense Layer 1
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Dense Layer 2
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Output Layer (4 Alzheimer Classes)
output = Dense(4, activation='softmax')(x)

# ---------------------------------------
# Build Model
# ---------------------------------------
model = Model(inputs=base_model.input, outputs=output)

# ---------------------------------------
# Loss Function with Label Smoothing
# ---------------------------------------
loss_function = CategoricalCrossentropy(label_smoothing=0.1)

# ---------------------------------------
# Compile Model
# ---------------------------------------
model.compile(
    optimizer=Adam(learning_rate=3e-5),
    loss=loss_function,
    metrics=['accuracy']
)

# ---------------------------------------
# Show Model Architecture
# ---------------------------------------
model.summary()