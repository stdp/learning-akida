import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected


MODEL_FBZ = "models/edge_learning_example.fbz"

HORSES = [
    "img/horse1.jpeg",  # The model will learn this image of a horse in Step1
    "img/horse2.jpeg",
    "img/horse3.jpeg",
    "img/horse4.jpeg",
    "img/horse5.jpeg",  # The script will test this image in Step2
    "img/horse6.jpeg",
]

DOGS = [
    "img/dog1.jpeg",  # The model will learn this image of a dog in Step1
    "img/dog2.jpg",
    "img/dog3.jpeg",
    "img/dog4.jpeg",
    "img/dog5.jpeg",  # The script will test this image in Step2
    "img/dog6.jpeg",
]

BATS = [
    "img/bat1.jpg",  # The model will learn this image of a bat in Step3
    "img/bat2.jpg",
    "img/bat3.jpg",  # The model will learn this image of a bat in Step3
    "img/bat4.jpg",
    "img/bat5.jpg",
    "img/bat6.jpg",  # The script will test this image of a bat in Step4
]

LABELS = {0: "unknown", 1: "Horse", 2: "Dog", 3: "Bat"}

NUM_CLASSES = 10
NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350

TARGET_WIDTH = 224
TARGET_HEIGHT = 224


"""
Change this step as you progress through this tutorial
"""

STEP = 1


"""

Step 1:

- Initialise a pretrained ImageNet model
- Remove the last layer of the ImageNet network
- Add a FullyConnected Akida layer as the last layer of the network
- Learn a 'Dog'

"""

if STEP == 1:

    ds, ds_info = tfds.load("coil100:2.*.*", split="train", with_info=True)
    model_keras = mobilenet_edge_imagenet_pretrained()

    # # Convert it to akida
    model_ak = convert(model_keras, input_scaling=(128, 128))

    # remove the last layer of network, replace with Akida learning layer
    model_ak.pop_layer()
    layer_fc = FullyConnected(
        name="akida_edge_layer",
        num_neurons=NUM_CLASSES * NUM_NEURONS_PER_CLASS,
        activations_enabled=False,
    )
    model_ak.add(layer_fc)
    model_ak.compile(
        num_weights=NUM_WEIGHTS, num_classes=NUM_CLASSES, learning_competition=0.1
    )

    # Learn a horse
    image = load_img(
        HORSES[0], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")
    model_ak.fit(input_arr, 1)

    # Learn a dog
    image = load_img(
        DOGS[2], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")
    model_ak.fit(input_arr, 2)

    # output the saved akida model
    model_file = os.path.join("", MODEL_FBZ)
    model_ak.save(model_file)


"""

Step 2:

- Test the network by trying to identify a HORSE and a DOG
* Spend some time understanding where you may need more than 1 shot, eg. dogs are weird looking
"""

if STEP == 2:

    model_ak = Model(filename=MODEL_FBZ)

    # image to test
    image = load_img(
        HORSES[3], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")

    predictions = model_ak.predict(input_arr, num_classes=NUM_CLASSES)

    print("should be class 1 (horse)", predictions[0])

    # image to test
    image = load_img(
        DOGS[5], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")

    predictions = model_ak.predict(input_arr, num_classes=NUM_CLASSES)

    print("should be class 2 (dog)", predictions[0])


"""

Step 3:

- Learn a new class, BAT, on output neuron 3
- Add another BAT image to strengthen the classification
- Save the updated Akida model

"""

if STEP == 3:

    model_ak = Model(filename=MODEL_FBZ)

    # Learn a bat
    image = load_img(
        BATS[0], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")
    model_ak.fit(input_arr, 3)

    # Learn another bat
    image = load_img(
        BATS[2], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")
    model_ak.fit(input_arr, 3)

    # output the saved akida model
    model_file = os.path.join("", MODEL_FBZ)
    model_ak.save(model_file)


"""

Step 4:

- Identify the BAT in random images using the LABELS dict

"""

if STEP == 4:

    model_ak = Model(filename=MODEL_FBZ)

    # recognise a bat
    image = load_img(
        BATS[5], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
    )
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr], dtype="uint8")

    predictions = model_ak.predict(input_arr, num_classes=NUM_CLASSES)

    print("the creature is a:", LABELS[predictions[0]])
