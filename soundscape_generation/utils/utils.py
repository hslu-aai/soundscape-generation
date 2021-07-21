import os
import tensorflow as tf
from datetime import datetime


def create_folder(path):
    """
    Creates a folder at a given path.
    :param path: the folder path to create.
    :return: the path of the newly created path.
    """
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def create_folder_for_experiment(model_name, dataset_name):
    # create experiment folder for current dataset
    experiment_path = create_folder(os.path.join(os.getcwd(), 'experiments'))
    experiment_path = create_folder(os.path.join(experiment_path, dataset_name))

    # create current experiment folder
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_folder_name = "{0}-{1}".format(model_name, current_time)
    experiment_path = create_folder(os.path.join(experiment_path, experiment_folder_name))
    print('Created experiment path at: {}'.format(experiment_path))

    return experiment_path


def set_gpu_experimental_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def normalize_image(img):  # map pixel intensities to float32 in [-1, 1]
    """
    Normalize a given image to float32 with range [-1, 1].
    :param img: image to normalize.
    :return: normalized image.
    """
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def denormalize_image(img):
    """
    Denormalize a given image that is float32 and in range [-1, 1].
    Map pixel intensities to uint8 (i.e. in range [0, 255]).
    :param img: image to denormalize.
    :return: denormalized image.
    """
    img = (img + 1.0) * 127.5
    img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.uint8)


def read_image(path, image_size):
    """
    Load the given image from a path and resize it.
    :param path: image to load.
    :param image_size: size to which the image should be resized to.
    :return: resized image.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3, dtype=tf.uint8)
    img = normalize_image(img)
    img = tf.image.resize(img, image_size, method='bilinear')  # (image_size[0], image_size[1], 3)
    return img


def read_segmentation(path, image_size, id2label):
    """
    Read segmentation (class label for each pixel) and resize it to image size.
    :param path: path to the segmentation image.
    :param image_size: size to which the segmentation mask should be resized to.
    :param id2label: ids of the segmentation mask.
    :return: resized segmentation mask.
    """
    seg = tf.io.read_file(path)
    seg = tf.image.decode_png(seg, channels=1, dtype=tf.uint8)
    # resize with 'nearest' method to avoid creating new classes
    seg = tf.image.resize(seg, image_size, method='nearest')
    seg = tf.squeeze(seg)
    seg = tf.gather(id2label, tf.cast(seg, tf.int32))  # (image_size[0], image_size[1])
    return seg
