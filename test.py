import gzip
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Load the "brain" you saved earlier
print("Waking up the model...")
model = load_model('mnist/my_mnist_model.keras')

# 2. Helper functions to load the test data
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Load ONLY the test images this time (the 10,000 images it hasn't memorized)
print("Loading test images...")
X_test = load_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz')

# Format the test data exactly how we formatted the training data
X_test_normalized = X_test.astype('float32') / 255.0

# 3. Pick a specific image to test! (Pick a number between 0 and 9999)
image_index = 42 
test_image = X_test_normalized[image_index]
true_label = y_test[image_index]

# 4. Ask the model to make a prediction
# The model expects a "batch" (a list) of images, so we wrap our single image in brackets
print("Asking the model to guess...")
prediction_array = model.predict(np.array([test_image]))

# The model outputs 10 probabilities (one for each digit 0-9). 
# np.argmax finds the highest probability, which is the model's final answer.
predicted_number = np.argmax(prediction_array)

print(f"\n--- RESULTS ---")
print(f"The actual number is: {true_label}")
print(f"The model guessed:    {predicted_number}")

# 5. Pop open a window to visually show the result
plt.imshow(X_test[image_index], cmap='gray')
plt.title(f"Model Guessed: {predicted_number} | Actual: {true_label}")
plt.axis('off') # Hides the grid numbers
plt.show()