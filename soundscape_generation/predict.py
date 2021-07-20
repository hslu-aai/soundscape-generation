import argparse
import glob
import heapq
import numpy as np
import os
import tensorflow as tf
import time
from PIL import Image
from operator import itemgetter
from soundscape_generation.models.ERFNet import ERFNet
from soundscape_generation.dataset.cityscapes import CityscapesDataset
from soundscape_generation.utils.utils import read_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

foreground_sounds_dict, background_sounds_dict = {}, {}
DEFAULT_MODEL = "experiments/Cityscapes/ERFNet-Pretrained/pretrained.h5"


def main(args):
    img_h_orig, img_w_orig = 1024, 2048  # original size of images in Cityscapes dataset
    img_h, img_w = args.img_height, args.img_width

    if not os.path.exists('extended_prototype/test_segmentations'):
        os.makedirs('extended_prototype/test_segmentations')
        print('test_segmentations directory created.')

    dataset = CityscapesDataset()

    print('Creating network and loading weights...')
    network = ERFNet(dataset.num_classes)

    # Initialize network weights
    inp_test = tf.random.normal(shape=(1, img_h, img_w, 3))
    out_test = network(inp_test, is_training=False)
    print('Shape of network\'s output:', out_test.shape)

    # Load weights and images from given paths
    if eval(args.weights) is None:
        args.weights = DEFAULT_MODEL
    weights_path = os.path.join(os.getcwd(), args.weights)
    image_paths = sorted(glob.glob(os.path.join(os.getcwd(), args.test_images, '*.{}'.format(args.test_images_type))))  # Specify Image file type

    network.load_weights(weights_path)
    print('Weights from {} loaded correctly.'.format(weights_path))

    inference_times = []
    for image_path in image_paths:
        t0 = time.time()

        image = read_image(image_path, (img_h, img_w))
        x = tf.expand_dims(image, axis=0)  # (1, img_h, img_w, 3)

        y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
        y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)  # (img_h, img_w)
        y_pred_colors = tf.gather(dataset.class_colors, y_pred_labels)
        y_pred_colors = tf.cast(y_pred_colors, tf.uint8)  # (img_h, img_w, 3)
        y_pred_colors = tf.image.resize(y_pred_colors, (img_h_orig, img_w_orig),
                                        method='nearest')  # (img_h_orig, img_w_orig, 3)
        t1 = time.time()

        # Save segmentation
        save_path = image_path.replace('.{}'.format(args.test_images_type), '_pred.{}'.format(args.test_images_type))
        segmentation = Image.fromarray(y_pred_colors.numpy())
        segmentation.save(save_path)

        print('Segmentation of image\n {}\nsaved in\n {}.'.format(image_path, save_path))
        inference_times.append(t1 - t0)

        # Print detected_objects
        _, tail = os.path.split(image_path)
        foreground_sounds, background_sounds = get_sounds(image_path, save_path, y_pred_labels, dataset)
        foreground_sounds_dict[tail] = foreground_sounds
        background_sounds_dict[tail] = background_sounds
    mean_inference_time = sum(inference_times) / len(inference_times)
    print('\nAverage inference time: {:.3f} s'.format(mean_inference_time))
    return foreground_sounds_dict, background_sounds_dict


def get_sounds(image_path, save_path, y_pred_labels, dataset):
    print('Prediction of image\n{}\n{}'.format(image_path, save_path))
    unique, counts = np.unique(y_pred_labels.numpy(), return_counts=True)
    detected_objects_dict = dict(zip(unique, counts))
    topitems = heapq.nlargest(1, detected_objects_dict.items(), key=itemgetter(1))
    background_sounds_dict = dict(topitems)
    background_sounds_array = np.fromiter(background_sounds_dict.keys(), dtype=int)
    detected_objects_array = np.fromiter(detected_objects_dict.keys(), dtype=int)
    foreground_sounds_array = [i for i in detected_objects_array if i not in background_sounds_array]
    detected_objects_names = []
    foreground_sounds = []
    background_sounds = []
    for i in range(len(detected_objects_array)):
        detected_objects_names.append(dataset.class_names[detected_objects_array[i]])
    for i in range(len(foreground_sounds_array)):
        foreground_sounds.append(dataset.class_names[foreground_sounds_array[i]])
    for i in range(len(background_sounds_array)):
        background_sounds.append(dataset.class_names[background_sounds_array[i]])
    print('Detected objects: {}'.format(detected_objects_names))
    print('Objects in background: {}'.format(background_sounds))
    print('Objects in foreground: {}'.format(foreground_sounds))
    return foreground_sounds, background_sounds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--weights', type=str, default="None", help='Relative path of network weights')
    parser.add_argument('--test_images', type=str, default="data/test_images/", help='Relative path of the test images')
    parser.add_argument('--test_images_type', type=str, default="jpg", help='Test image types')

    args = parser.parse_args()
    main(args)
