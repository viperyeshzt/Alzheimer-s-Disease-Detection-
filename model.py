from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load Pretrained EfficientNetB0
# -----------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# -----------------------------
# Freeze most layers
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# -----------------------------
# Custom Classification Head
# -----------------------------
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

output = Dense(4, activation='softmax')(x)

# -----------------------------
# Final Model
# -----------------------------
model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show architecture
model.summary()