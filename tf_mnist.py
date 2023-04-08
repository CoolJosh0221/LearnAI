# Import necessary modules
import tensorflow as tf
from keras import layers
from tensorflow import keras

# Load the MNIST dataset
(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.mnist.load_data()

# Build the model
model = keras.Sequential(
    [layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")]
)

# Compile the model
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Preprocess the data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=124)

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:", test_acc)

# Make predictions
test_digits = test_images[:10]
predictions = model.predict(test_digits)
print(predictions[0])

# Get a batch of data for further processing (optional)
batch = train_images[128:256]
