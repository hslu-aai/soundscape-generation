import glob
import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

from soundscape_generation.utils.utils import read_image, read_segmentation


class CityscapesDataset:
    """
    Class for the Cityscapes Dataset.
    https://www.cityscapes-dataset.com/
    TODO: create tf.DataSet for it
    """
    # name of the dataset
    name = 'Cityscapes'

    # labels of the dataset
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',  # 0-5
                   'traffic light', 'traffic sign', 'vegetation', 'terrain',  # 6-9
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',  # 10-16
                   'motorcycle', 'bicycle', 'other']  # 17-19 class 'other' includes the rest of classes

    # as ids are used in the 'labelIds' annotation images, we need the class label (0 to 19) for every id (0 to 34)
    # this is needed since there are additional ids that all correspond to the same class label
    id2label = tf.constant([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3,
                            4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12,
                            13, 14, 15, 19, 19, 16, 17, 18, 19], tf.int32)

    # colors for each class in the dataset, used for visualizing the predicted mask
    class_colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
                    (0, 0, 230), (119, 11, 32), (0, 0, 0)]

    def __init__(self, image_size):
        """
        Initializes the dataset.
        """
        # set variables
        self.image_size = image_size

        # load images train
        img_path = os.path.join(os.getcwd(), 'data', 'images', 'train', '*', '*.png')
        seg_path = os.path.join(os.getcwd(), 'data', 'segmentations', 'train', '*', '*labelIds.png')
        self.image_paths = sorted(glob.glob(img_path))
        self.segmentation_paths = sorted(glob.glob(seg_path))
        self.num_images = len(self.image_paths)

        # load images validation
        val_img_path = os.path.join(os.getcwd(), 'data', 'images', 'val', '*', '*.png')
        val_seg_path = os.path.join(os.getcwd(), 'data', 'segmentations', 'val', '*', '*labelIds.png')
        self.val_image_paths = sorted(glob.glob(val_img_path))
        self.val_segmentation_paths = sorted(glob.glob(val_seg_path))
        self.num_val_images = len(self.val_image_paths)

        # convert the class names to labels and create a mapping dict from name to label_id
        self.name2label = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Weight for every class (used to compute the loss). Under-represented classes have a larger weight.
        self.class_weights = self.compute_class_weights()

    def __get_batch(self, batch_id, batch_size) -> Tuple[np.array, np.array]:
        """
        Returns a training batch of the dataset.
        :param batch_id: index of the batch to return.
        :param batch_size: size of the batch to return.
        :return: tuple of the batch [images, labels].
        """
        start_id = batch_id * batch_size
        end_id = start_id + batch_size

        batch_images = [read_image(self.image_paths[i], self.image_size) for i in range(start_id, end_id)]
        batch_segmentations = [read_segmentation(self.segmentation_paths[i], self.image_size, self.id2label) for i
                               in range(start_id, end_id)]
        return batch_images, batch_segmentations

    def get_training_batch(self, batch_id, batch_size) -> Tuple[np.array, np.array]:
        """
        Returns a training batch of the dataset.
        :param batch_id: index of the batch to return.
        :param batch_size: size of the batch to return.
        :return: tuple of the training batch [images, labels].
        """
        batch_images, batch_segmentations = self.__get_batch(batch_id, batch_size)

        # Data augmentation
        for i in range(batch_size):
            if random.random() > 0.5:  # horizontal flip with probability 0.5
                batch_images[i] = tf.reverse(batch_images[i], axis=[1])
                batch_segmentations[i] = tf.reverse(batch_segmentations[i], axis=[1])
            batch_images[i] = tf.image.random_brightness(batch_images[i], max_delta=0.08)  # random brightness
            batch_images[i] = tf.image.random_contrast(batch_images[i], lower=0.95, upper=1.05)  # random contrast

        x = tf.stack(batch_images, axis=0)  # (batch_size, img_h, img_w, 3)
        y_true_labels = tf.stack(batch_segmentations, axis=0)  # (batch_size, img_h, img_w)
        return x, y_true_labels

    def get_validation_batch(self, batch_id, val_batch_size) -> Tuple[np.array, np.array]:
        """
        Returns a validation batch of the dataset.
        :param batch_id: index of the batch to return.
        :param val_batch_size: size of the batch to return.
        :return: tuple of the validation batch [images, labels].
        """
        batch_images, batch_segmentations = self.__get_batch(batch_id, val_batch_size)

        x = tf.stack(batch_images, axis=0)  # (batch_size, img_h, img_w, 3)
        y_true_labels = tf.stack(batch_segmentations, axis=0)  # (batch_size, img_h, img_w)
        return x, y_true_labels

    def shuffle_training_paths(self):
        """
        Shuffles the training images paths of the image and segmentation in union.
        """
        aux = list(zip(self.image_paths, self.segmentation_paths))
        random.shuffle(aux)
        self.image_paths, self.segmentation_paths = zip(*aux)

    def compute_class_weights(self):
        """
        Computes the class weights for the dataset.
        :return: the class weights.
        """
        num_pixels_per_class = [0] * self.num_classes  # store the number of pixels for each class

        for i, segmentation_path in enumerate(self.segmentation_paths):
            segmentation = read_segmentation(segmentation_path, self.image_size, self.id2label)
            for class_label in range(self.num_classes):
                num_pixels = tf.reduce_sum(tf.cast(tf.equal(segmentation, class_label), tf.int32))
                num_pixels_per_class[class_label] += num_pixels

        class_probs = tf.divide(num_pixels_per_class[:-1], tf.reduce_sum(num_pixels_per_class[:-1]))
        class_weights = 1 / tf.math.log(1.1 + class_probs)
        # class 'other' has weight 0 in order to disregard its pixels
        class_weights = np.append(class_weights.numpy(), 0.0)
        return class_weights
