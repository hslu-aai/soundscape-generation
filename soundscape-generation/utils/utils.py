import tensorflow as tf


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
