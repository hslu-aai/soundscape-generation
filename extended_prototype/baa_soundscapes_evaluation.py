import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import glob
import tensorflow as tf
import numpy as np

from datasets import CityscapesDataset
from models import ERFNet
from evaluation import evaluate

from utils import read_image



def main(args):

    img_h, img_w = args.img_height, args.img_width
    val_batch_size = args.val_batch_size
    is_validation_set = args.is_validation_set

    if (is_validation_set):
        image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'test_images', '*.png')))
    else:
        image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'test_images', '*.jpg')))

    own_test_set_true = [
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0], 
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0], 
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 1], 
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    ]

    dataset = CityscapesDataset()

    print('Creating network and loading weights...')
    network = ERFNet(dataset.num_classes)

    # Initialize network weights
    inp_test = tf.random.normal(shape=(1, img_h, img_w, 3))
    out_test = network(inp_test, is_training=False)
    print('Shape of network\'s output:', out_test.shape)

    # Load weights and images from given paths
    weights_path = os.path.join(os.getcwd(), args.weights)
    network.load_weights(weights_path)
    print('Weights from {} loaded correctly.'.format(weights_path))
    #iou_per_class, iou_mean = evaluate(dataset, network, val_batch_size, (img_h, img_w))
    #print("iou_per_class: {}, iou_mean: {}".format(iou_per_class, iou_mean))
    get_total_percision(dataset, network, val_batch_size, (img_h, img_w), is_validation_set, own_test_set_true, image_paths)
    get_total_recall(dataset, network, val_batch_size, (img_h, img_w), is_validation_set, own_test_set_true, image_paths)

def get_total_percision(dataset, network, val_batch_size, image_size, is_validation_set, own_test_set_true, image_paths):
    total_tp = tf.zeros((1), tf.int64)
    total_tp_and_fp = tf.zeros((1), tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    if(is_validation_set):
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fp_batch = get_precisition_in_batch(y_true_labels, y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fp += tp_and_fp_batch
            batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
            print('Precistion from batch {} / {} is {}.'.format(batch+1, num_val_batches, batchprecision))
            print()
        total_set_precision = tf.divide(total_tp, total_tp_and_fp)
        print('Total Precistion on validation set is {}'.format(total_set_precision))
        return total_set_precision
    else:
        print()
        test_set_true_counter = 0
        for image_path in image_paths:
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fp_batch = get_precisition_in_batch(np.array(own_test_set_true[test_set_true_counter]), y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fp += tp_and_fp_batch
            batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
            test_set_true_counter += 1
            print('Precistion from image {}: {}.'.format(image_path, batchprecision))
            print()
        total_set_precision = tf.divide(total_tp, total_tp_and_fp)
        print('Total Precistion on own test set is {}'.format(total_set_precision))
        return total_set_precision


def get_precisition_in_batch(y_true, y_pred, num_classes, is_validation_set):
    tp_batch, tp_and_fp_batch = [], []
    tp = 0
    fp = 0
    if (is_validation_set):
        y_true_labels_list = []
        y_pred_labels_list = []
        for class_label in range(num_classes-1):
            true_equal_class = tf.cast(tf.equal(y_true, class_label), tf.int32)
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_true_labels_list.append(tf.reduce_sum(true_equal_class))
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_true_array = np.array(y_true_labels_list)
        y_pred_array = np.array(y_pred_labels_list)
        
        # Convert to bool arrays
        y_true_bool_array = np.where(y_true_array > 0, 1, 0)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)
        
        for i in range(len(y_true_bool_array)):
            if(y_true_bool_array[i] ==1 and y_pred_bool_array[i]==1):
                tp += 1
            if(y_pred_bool_array[i] > y_true_bool_array[i]):
                fp += 1
        print("y_true: {}".format(y_true_bool_array))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Positives: {}".format(tp, fp))
    else:
        y_pred_labels_list = []

        for class_label in range(num_classes-1):
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_pred_array = np.array(y_pred_labels_list)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)
        for i in range(len(y_pred_bool_array)):
            if(y_true[i]==1 and y_pred_bool_array[i]==1):
                tp += 1
            if(y_pred_bool_array[i] > y_true[i]):
                fp += 1
        print("y_true: {}".format(y_true))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Positives: {}".format(tp, fp))
    tp_batch.append(tp)
    tp_and_fp_batch.append(tp + fp)
    return tp_batch, tp_and_fp_batch     

def get_total_recall(dataset, network, val_batch_size, image_size, is_validation_set, own_test_set_true, image_paths):
    total_tp = tf.zeros((1), tf.int64)
    total_tp_and_fn = tf.zeros((1), tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    if(is_validation_set):
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch = get_recall_in_batch(y_true_labels, y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fn += tp_and_fn_batch
            batchrecall= tf.divide(tp_batch, tp_and_fn_batch)
            print('Recall from batch {} / {} is {}.'.format(batch+1, num_val_batches, batchrecall))
            print()
        total_set_recall = tf.divide(total_tp, total_tp_and_fn)
        print('Total Recall on validation set is {}'.format(total_set_recall))
        return total_set_recall
    else:
        print()
        test_set_true_counter = 0
        for image_path in image_paths:
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch = get_recall_in_batch(np.array(own_test_set_true[test_set_true_counter]), y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fn += tp_and_fn_batch
            batchrecall= tf.divide(tp_batch, tp_and_fn_batch)
            test_set_true_counter += 1
            print('Recall from image {}: {}.'.format(image_path, batchrecall))
            print()
        total_set_recall = tf.divide(total_tp, total_tp_and_fn)
        print('Total Recall on own test set is {}'.format(total_set_recall))
        return total_set_recall

def get_recall_in_batch(y_true, y_pred, num_classes, is_validation_set):

    tp_batch, tp_and_fn_batch = [], []
    tp = 0
    fn = 0
    if (is_validation_set):
        y_true_labels_list = []
        y_pred_labels_list = []
        for class_label in range(num_classes-1):
            true_equal_class = tf.cast(tf.equal(y_true, class_label), tf.int32)
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_true_labels_list.append(tf.reduce_sum(true_equal_class))
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_true_array = np.array(y_true_labels_list)
        y_pred_array = np.array(y_pred_labels_list)
        
        # Convert to bool arrays
        y_true_bool_array = np.where(y_true_array > 0, 1, 0)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)

        for i in range(len(y_true_bool_array)):
            if(y_true_bool_array[i]==1 and y_pred_bool_array[i]==1):
                tp += 1
            if(y_pred_bool_array[i] < y_true_bool_array[i]):
                fn += 1
        print("y_true: {}".format(y_true_bool_array))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}".format(tp, fn))
    else:
        y_pred_labels_list = []
        for class_label in range(num_classes-1):
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_pred_array = np.array(y_pred_labels_list)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)

        for i in range(len(y_pred_labels_list)):
            if(y_true[i]==1 and y_pred_bool_array[i]==1):
                tp += 1
            if(y_pred_bool_array[i] < y_true[i]):
                fn += 1
        print("y_true: {}".format(y_true))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}".format(tp, fn))
    tp_batch.append(tp)
    tp_and_fn_batch.append(tp + fn)
    return tp_batch, tp_and_fn_batch

if __name__ == '__main__':
    os.chdir("extended_prototype")

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_validation_set', type=bool, default=False, help='Evaluation on validation or own dataset')
    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--weights', type=str, default="pretrained/pretrained.h5", help='Relative path of network weights')
    args = parser.parse_args()
    
    main(args)
   
    