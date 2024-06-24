# Image Recognition Model

This repository contains code for training and deploying an image recognition model using the CIFAR-10 dataset. The model is built using TensorFlow and Keras.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)
- [Dataset](#dataset)

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/image-recognition-model.git
cd image-recognition-model
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Model Training

The model can be trained by running the training script. Note that this step is commented out in the provided code for convenience.

```python
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images = training_images / 255.0
testing_images = testing_images / 255.0

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

model.save('my_model.keras')
```

### Model Evaluation

After training, evaluate the model to check its performance.

```python
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
```

### Prediction

Load the trained model and make predictions on new images.

```python
model = models.load_model('my_model.keras')

img = cv.imread('./imagescale/resized_plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')
plt.show()
```

## Dataset

The CIFAR-10 dataset is used for training and evaluating the model. It consists of 60,000 32x32 color images in 10 different classes.mage Recognition Model
This repository contains code for training and deploying an image recognition model using the CIFAR-10 dataset. The model is built using TensorFlow and Keras.

Table of Contents
Installation
Usage
Model Training
Model Evaluation
Prediction
Dataset
License
Installation
Clone the repository:

bash
Copy code
git clone <https://github.com/your-username/image-recognition-model.git>
cd image-recognition-model
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Model Training:

The model can be trained by running the training script. Note that this step is commented out in the provided code for convenience.

python
Copy code
training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images = training_images / 255.0
testing_images = testing_images / 255.0

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

model.save('my_model.keras')
Model Evaluation:

After training, evaluate the model to check its performance.

python
Copy code
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
Prediction:

Load the trained model and make predictions on new images.

python
Copy code
model = models.load_model('my_model.keras')

img = cv.imread('./imagescale/resized_plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')
plt.show()
Dataset
The CIFAR-10 dataset is used for training and evaluating the model. It consists of 60,000 32x32 color images in 10 different classes.
