import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv

from PIL import Image


def save_images(images, labels, split_dir):
    for i, (image, label) in enumerate(zip(images, labels)):
        class_name = class_names[label[0]]

        img_path = os.path.join(split_dir, class_name, f"{class_name}_{i}.png")

        Image.fromarray(image).save(img_path)

def save_labels(labels, split_dir, split_name):
    with open(os.path.join(split_dir, f"{split_name}_labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "class_name"])
        for i, label in enumerate(labels):
            class_name = class_names[label[0]]
            filename = f"{class_name}_{i}.png"
            writer.writerow([filename, label[0], class_name])

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    base_dir = "cifar10_dataset"
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    for split_dir in [train_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    save_images(x_train, y_train, train_dir)
    save_images(x_test, y_test, test_dir)

    save_labels(y_train, train_dir, "train")
    save_labels(y_test, test_dir, "test")
