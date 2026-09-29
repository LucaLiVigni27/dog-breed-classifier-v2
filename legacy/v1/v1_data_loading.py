

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# create ImageDataGenerator for train and validation splits
fulldata = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# training dataset (80% of data)
train_dataset = fulldata.flow_from_directory(
    "/Users/lucalivigni/Downloads/NSDC\ PROJECT\ ",
    target_size=(200, 200),
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# validation dataset (20% of data)
validation_dataset = fulldata.flow_from_directory(
    "/Users/lucalivigni/Downloads/NSDC\ PROJECT\  ",
    target_size=(200, 200),
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# train image labels
train_labels = {value: key for key, value in train_dataset.class_indices.items()}
len(train_labels)

# validation image labels
validation_labels = {value: key for key, value in validation_dataset.class_indices.items()}
len(validation_labels)

# view sample training images
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
idx = 0

for i in range(2):
    for j in range(5):
        label = train_labels[np.argmax(train_dataset[0][1][idx])]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(train_dataset[0][0][idx][:, :, :])
        ax[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.suptitle("Sample Training Images", fontsize=21)
plt.show()
