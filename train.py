import gzip
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Helper functions to read those specific binary .gz files
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

print("Loading local MNIST data...")
X_train = load_mnist_images('mnist/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('mnist/train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz')

# 2. Format the data (Normalize pixels to be between 0 and 1)
print("Formatting data...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert the labels into a format the network understands (One-Hot Encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Build the Neural Network
print("Building the model...")
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model AND save the history!
print("Starting training...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save the brain
model.save('mnist/my_mnist_model.keras')
print("Model saved to disk!")

# 5. DRAW THE CURVES!
print("Generating training curves...")
plt.figure(figsize=(12, 5))

# Graph 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
plt.title('Model Accuracy over 5 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Graph 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
plt.title('Model Loss over 5 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (Error)')
plt.legend()
plt.grid(True)

plt.savefig('mnist/training_curves.png')
plt.show()