import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# 1. DATASET PATHS
train_dir = "dataset/train"
validation_dir = "dataset/validation"
test_dir = "dataset/test"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32


# 2. DATA GENERATORS

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

# Test data
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# 3. LOAD TRAINING DATA

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# 4. LOAD VALIDATION DATA

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# Display class mapping
print("\nClass indices:")
print(train_generator.class_indices)


# 5. BASIC CNN MODEL

model = keras.Sequential([

    keras.Input(shape=(160, 160, 3)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Flatten(),

    layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# 6. COMPILE CNN
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 7. TRAIN BASIC CNN

print("\n==============================")
print("TRAINING BASIC CNN")
print("==============================")

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)


# 8. PLOT TRAINING AND VALIDATION ACCURACY
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.figure(figsize=(8, 5))

plt.plot(
    acc,
    label="Training Accuracy"
)

plt.plot(
    val_acc,
    label="Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.show()

# 9. PREDICT AN IMAGE USING BASIC CNN

def predict_image(img_path):

    if not os.path.exists(img_path):

        print("\nImage not found:")
        print(img_path)

        return

    # Load image
    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Normalize
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = prediction[0][0]

    # Determine class
    if probability > 0.5:

        label = "Dog"
        confidence = probability * 100

    else:

        label = "Cat"
        confidence = (1 - probability) * 100

    # Display result
    plt.figure(figsize=(6, 6))

    plt.imshow(
        image.load_img(img_path)
    )

    plt.title(
        f"Prediction: {label}\nConfidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.show()

    print("\nPrediction:", label)
    print("Confidence:", f"{confidence:.2f}%")


# 10. TEST ONE IMAGE WITH BASIC CNN

predict_image(
    "exp7data/cat2.jpeg"
)


# 11. SAVE BASIC CNN MODEL
model.save(
    "dog_cat_classifier.keras"
)

print("\nBasic CNN model saved as:")
print("dog_cat_classifier.keras")



# 12. MOBILE NET V2 TRANSFER LEARNING



print("MOBILENETV2 TRANSFER LEARNING")



base_model = tf.keras.applications.MobileNetV2(

    input_shape=(160, 160, 3),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers
base_model.trainable = False



# 13. CREATE MOBILENETV2 MODEL

model = keras.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])



# 14. COMPILE MOBILENETV2
model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# 15. TRAIN MOBILENETV2
history = model.fit(

    train_generator,

    epochs=5,

    validation_data=validation_generator
)



# 16. LOAD TEST DATA
print("LOADING TEST DATA")

if not os.path.exists(test_dir):

    print("\nERROR: Test folder not found.")
    print("Expected folder:")
    print("dataset/test")

else:

    test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="binary",

        shuffle=False
    )

    # 17. CHECK TEST DATA
    if test_generator.samples == 0:

        print("\nERROR: No test images found.")

        print("\nYour test folder must be arranged like this:")

        print("dataset/test/cat/")
        print("dataset/test/dog/")

    else:

        print("\nTest images found:",
              test_generator.samples)

        print(
            "Test class indices:",
            test_generator.class_indices
        )

        # 18. EVALUATE MOBILENETV2
             
        print("TESTING MOBILENETV2")
     
        test_loss, test_acc = model.evaluate(
            test_generator
        )

        # 19. DISPLAY TEST RESULTS
   
        print("TEST RESULTS")
       
        print(
            "Test Accuracy:",
            f"{test_acc * 100:.2f}%"
        )

        print(
            "Test Loss:",
            f"{test_loss:.4f}"
        )