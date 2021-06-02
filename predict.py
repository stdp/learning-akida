import numpy as np
from akida import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_FBZ = "models/edge_learning_example.fbz"

IMGS = [
    "img/horse1.jpeg",
    "img/horse2.jpeg",
    "img/horse3.jpeg",
    "img/horse4.jpeg",
    "img/horse5.jpeg",
    "img/horse6.jpeg",
    "img/dog1.jpeg",
    "img/dog2.jpg",
    "img/dog3.jpeg",
    "img/dog4.jpeg",
    "img/dog5.jpeg",
    "img/dog6.jpeg",
    "img/bat1.jpg",
    "img/bat2.jpg",
    "img/bat3.jpg",
    "img/bat4.jpg",
    "img/bat5.jpg",
    "img/bat6.jpg",
]

LABELS = {0: "unknown", 1: "Horse", 2: "Dog", 3: "Bat"}

NUM_CLASSES = 10
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

IMG_TO_TEST = 15  # which image do you want to test?

model_ak = Model(filename=MODEL_FBZ)

# load an image to test
image = load_img(
    IMGS[IMG_TO_TEST], target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
)
input_arr = img_to_array(image)
input_arr = np.array([input_arr], dtype="uint8")

# identify the creature
predictions = model_ak.predict(input_arr, num_classes=NUM_CLASSES)

print("the creature is a: {}".format(LABELS[predictions[0]]))
