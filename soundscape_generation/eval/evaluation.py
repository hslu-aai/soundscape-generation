import numpy as np
import tensorflow as tf

from soundscape_generation.utils.utils import read_image


def compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes):
    # y_true_labels: (val_batch_size, img_h, img_w)
    # y_pred_labels: (val_batch_size, img_h, img_w)

    batch_intersection, batch_union = [], []  # for each class, store the sum of intersections and unions in the batch

    for class_label in range(num_classes - 1):  # ignore class 'other'
        true_equal_class = tf.cast(tf.equal(y_true_labels, class_label), tf.int32)
        pred_equal_class = tf.cast(tf.equal(y_pred_labels, class_label), tf.int32)

        intersection = tf.reduce_sum(tf.multiply(true_equal_class, pred_equal_class))  # TP (true positives)
        union = tf.reduce_sum(true_equal_class) + tf.reduce_sum(
            pred_equal_class) - intersection  # TP + FP + FN = (TP + FP) + (TP + FN) - TP

        batch_intersection.append(intersection)
        batch_union.append(union)

    return tf.cast(tf.stack(batch_intersection, axis=0), tf.int64), tf.cast(tf.stack(batch_union, axis=0),
                                                                            tf.int64)  # (19,)


def evaluate(dataset, network, val_batch_size, image_size):
    # Compute IoU on validation set (IoU = Intersection / Union)

    total_intersection = tf.zeros((19), tf.int64)
    total_union = tf.zeros((19), tf.int64)

    print('Evaluating on validation set...')
    num_val_batches = dataset.num_val_images // val_batch_size
    for batch in range(num_val_batches):
        x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)

        y_pred_logits = network(x, is_training=False)
        y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)

        batch_intersection, batch_union = compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels,
                                                                                  dataset.num_classes)
        total_intersection += batch_intersection
        total_union += batch_union

    iou_per_class = tf.divide(total_intersection, total_union)  # IoU for each of the 19 classes
    iou_mean = tf.reduce_mean(iou_per_class)  # Mean IoU over the 19 classes

    return iou_per_class, iou_mean


def get_total_percision(dataset, network, val_batch_size, image_size, is_validation_set, own_test_set_true,
                        image_paths):
    total_tp = tf.zeros((1), tf.int64)
    total_tp_and_fp = tf.zeros((1), tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    if (is_validation_set):
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fp_batch = get_precisition_in_batch(y_true_labels, y_pred_labels, dataset.num_classes,
                                                                 is_validation_set)
            total_tp += tp_batch
            total_tp_and_fp += tp_and_fp_batch
            batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
            print('Precistion from batch {} / {} is {}.'.format(batch + 1, num_val_batches, batchprecision))
            print()
        total_set_precision = tf.divide(total_tp, total_tp_and_fp)
        print('Total Precistion on validation set is {}'.format(total_set_precision))
        return total_set_precision
    else:
        test_set_true_counter = 0
        for image_path in image_paths:
            print('-'*20 + image_path + '-'*20)
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fp_batch = get_precisition_in_batch(np.array(own_test_set_true[test_set_true_counter]),
                                                                 y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fp += tp_and_fp_batch
            batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
            test_set_true_counter += 1
            print('Precistion: {}'.format(batchprecision))
        total_set_precision = tf.divide(total_tp, total_tp_and_fp)
        print('-'*20 + 'TOTAL PRECISION' + '-'*20)
        print('Total Precistion on own test set is {}'.format(total_set_precision))
        return total_set_precision


def get_precisition_in_batch(y_true, y_pred, num_classes, is_validation_set):
    tp_batch, tp_and_fp_batch = [], []
    tp = 0
    fp = 0
    if (is_validation_set):
        y_true_labels_list = []
        y_pred_labels_list = []
        for class_label in range(num_classes - 1):
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
            if (y_true_bool_array[i] == 1 and y_pred_bool_array[i] == 1):
                tp += 1
            if (y_pred_bool_array[i] > y_true_bool_array[i]):
                fp += 1
        print("y_true: {}".format(y_true_bool_array))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Positives: {}".format(tp, fp))
    else:
        y_pred_labels_list = []

        for class_label in range(num_classes - 1):
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_pred_array = np.array(y_pred_labels_list)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)
        for i in range(len(y_pred_bool_array)):
            if (y_true[i] == 1 and y_pred_bool_array[i] == 1):
                tp += 1
            if (y_pred_bool_array[i] > y_true[i]):
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
    if (is_validation_set):
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch = get_recall_in_batch(y_true_labels, y_pred_labels, dataset.num_classes,
                                                            is_validation_set)
            total_tp += tp_batch
            total_tp_and_fn += tp_and_fn_batch
            batchrecall = tf.divide(tp_batch, tp_and_fn_batch)
            print('Recall from batch {} / {} is {}.'.format(batch + 1, num_val_batches, batchrecall))
            print()
        total_set_recall = tf.divide(total_tp, total_tp_and_fn)
        print('Total Recall on validation set is {}'.format(total_set_recall))
        return total_set_recall
    else:
        test_set_true_counter = 0
        for image_path in image_paths:
            print('-'*20 + image_path + '-'*20)
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch = get_recall_in_batch(np.array(own_test_set_true[test_set_true_counter]),
                                                            y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fn += tp_and_fn_batch
            batchrecall = tf.divide(tp_batch, tp_and_fn_batch)
            test_set_true_counter += 1
            print('Recall: {}'.format(batchrecall))
        total_set_recall = tf.divide(total_tp, total_tp_and_fn)
        print('-'*20 + 'TOTAL RECALL' + '-'*20)
        print('Total Recall on own test set is {}'.format(total_set_recall))
        return total_set_recall


def get_recall_in_batch(y_true, y_pred, num_classes, is_validation_set):
    tp_batch, tp_and_fn_batch = [], []
    tp = 0
    fn = 0
    if (is_validation_set):
        y_true_labels_list = []
        y_pred_labels_list = []
        for class_label in range(num_classes - 1):
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
            if (y_true_bool_array[i] == 1 and y_pred_bool_array[i] == 1):
                tp += 1
            if (y_pred_bool_array[i] < y_true_bool_array[i]):
                fn += 1
        print("y_true: {}".format(y_true_bool_array))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}".format(tp, fn))
    else:
        y_pred_labels_list = []
        for class_label in range(num_classes - 1):
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_pred_array = np.array(y_pred_labels_list)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)

        for i in range(len(y_pred_labels_list)):
            if (y_true[i] == 1 and y_pred_bool_array[i] == 1):
                tp += 1
            if (y_pred_bool_array[i] < y_true[i]):
                fn += 1
        print("y_true: {}".format(y_true))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}".format(tp, fn))
    tp_batch.append(tp)
    tp_and_fn_batch.append(tp + fn)
    return tp_batch, tp_and_fn_batch
