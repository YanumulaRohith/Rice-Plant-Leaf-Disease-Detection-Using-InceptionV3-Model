import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Replace these paths with the paths to your full dataset
full_data_dir = r'full data'
train_data_dir = r'train_data_dir'
val_data_dir = r'val_data_dir'
test_data_dir = r'test_data_dir'

input_shape = (224, 224)  # Input image dimensions for InceptionV3

# Create directories for train, validation, and test data
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)

# Move a percentage of the full dataset to the train_data_dir
train_split_percentage = 0.7
val_split_percentage = 0.15
test_split_percentage = 0.15

for class_name in os.listdir(full_data_dir):
    class_dir = os.path.join(full_data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    images = os.listdir(class_dir)
    random.shuffle(images)
    total_images = len(images)
    train_split_idx = int(train_split_percentage * total_images)
    val_split_idx = int((train_split_percentage + val_split_percentage) * total_images)
    
    for i, image in enumerate(images):
        src_path = os.path.join(class_dir, image)
        if i < train_split_idx:
            dst_dir = os.path.join(train_data_dir, class_name)
        elif i < val_split_idx:
            dst_dir = os.path.join(val_data_dir, class_name)
        else:
            dst_dir = os.path.join(test_data_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, image)
        shutil.copy(src_path, dst_path)

# Continue with building the model and training

# Define the number of classes based on the folders in train_data_dir
num_classes = len(os.listdir(train_data_dir))

# Load InceptionV3 with pre-trained ImageNet weights (exclude top layer)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for rice plant disease classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define data generators for training, validation, and testing
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_accuracy}")

# Plot training history
plt.plot(np.arange(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
