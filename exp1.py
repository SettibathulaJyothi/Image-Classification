# Import Required Libraries
import tensorflow as tf  # TensorFlow is the backend framework 
from keras.models import Sequential  # Used to build a linear stack of layers
from keras.layers import Dense, Flatten  # Dense (Fully connected layer), Flatten (Convert 2D to 1D)
from keras.datasets import mnist  # Import the MNIST dataset 
from keras.utils import to_categorical  # Converts labels to one-hot encoding 
import numpy as np 
import matplotlib.pyplot as plt 

# Load and Preprocess the Data 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
train_images, test_images = train_images / 255.0, test_images / 255.0 
train_labels = to_categorical(train_labels, 10) 
test_labels = to_categorical(test_labels, 10)

# Build the MLP Model 
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a 1D array 
    Dense(128, activation="relu"),  # Fully connected layer with 128 neurons 
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
])

# Compile the Model 
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model 
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the Model 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_acc}')

# Make Predictions 
predictions = model.predict(test_images[:5]) 

# Plot the images with predicted labels 
for i in range(5):
    plt.imshow(test_images[i], cmap="gray") 
    plt.title(f'Predicted: {np.argmax(predictions[i])}')
    plt.show()