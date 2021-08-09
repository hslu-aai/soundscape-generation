import numpy as np
import tensorflow as tf

from soundscape_generation.utils.utils import read_image


def compute_intersection_and_union(y_true_labels, y_pred_labels, num_classes):
    """
    Computes the Intersection and union score for the given batch.
    :param y_true_labels: true labels of the batch (val_batch_size, img_h, img_w)
    :param y_pred_labels: predicted labels of the batch (val_batch_size, img_h, img_w)
    :param num_classes: number of in the dataset
    :return: the intersection and union score for the batch.
    """

    # for each class, store the sum of intersections and unions in the batch
    batch_intersection, batch_union = [], []

    # -1 ignores the class `other`
    for class_label in range(num_classes - 1):
        true_equal_class = tf.cast(tf.equal(y_true_labels, class_label), tf.int32)
        pred_equal_class = tf.cast(tf.equal(y_pred_labels, class_label), tf.int32)

        # TP (true positives)
        intersection = tf.reduce_sum(tf.multiply(true_equal_class, pred_equal_class))
        # TP + FP + FN = (TP + FP) + (TP + FN) - TP
        union = tf.reduce_sum(true_equal_class) + tf.reduce_sum(pred_equal_class) - intersection

        batch_intersection.append(intersection)
        batch_union.append(union)

    return tf.cast(tf.stack(batch_intersection, axis=0), tf.int64), \
           tf.cast(tf.stack(batch_union, axis=0), tf.int64)  # (19,)


def compute_iou(dataset, model, val_batch_size, image_size):
    """
    Computes the IoU (intersection over union) score on the validation set.
    :param dataset: dataset from which to get the validation set.
    :param model: the model to evaluate.
    :param val_batch_size: size of the validation batch.
    :param image_size: size of the images.
    :return: the IoU score per class and the mean over all classes.
    """

    total_intersection = tf.zeros(19, tf.int64)
    total_union = tf.zeros(19, tf.int64)

    print('Evaluating on validation set...')
    num_val_batches = dataset.num_val_images // val_batch_size
    for batch in range(num_val_batches):
        # get the validation batch
        X_val, y_val_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)

        # compute the predictions of the validation set
        y_val_pred_logits = model(X_val, is_training=False)
        y_val_pred_labels = tf.math.argmax(y_val_pred_logits, axis=-1, output_type=tf.int32)

        # compute the intersection and union scores
        batch_intersection, batch_union = compute_intersection_and_union(y_val_true_labels, y_val_pred_labels,
                                                                         dataset.num_classes)
        total_intersection += batch_intersection
        total_union += batch_union

    # IoU for each of the 19 classes
    iou_per_class = tf.divide(total_intersection, total_union)
    # Mean IoU over the 19 classes
    iou_mean = tf.reduce_mean(iou_per_class)

    return iou_per_class, iou_mean


def get_total_precision(dataset, network, val_batch_size, image_size, is_validation_set, own_test_set_true, image_paths):
    """
    Computes the total precision of the network on a given dataset.
    :param dataset: the dataset to evaluate on.
    :param network: the network to evaluate.
    :param val_batch_size: the size of the validation batch.
    :param image_size: the size of the image.
    :param is_validation_set: if the network should be evaluated on the validation set or not.
    :param own_test_set_true: if the network should be evaluated on a custom test set.
    :param image_paths: path of the images for the evaluation (only needed if the network is evaluated on a custom set).
    """
    total_tp = tf.zeros(1, tf.int64)
    total_tp_and_fp = tf.zeros(1, tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    if is_validation_set:
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, _, tp_and_fp_batch = get_confusion_matrix_batch(y_true_labels, y_pred_labels, dataset.num_classes,
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
            print('-' * 20 + image_path + '-' * 20)
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, _, tp_and_fp_batch = get_confusion_matrix_batch(
                np.array(own_test_set_true[test_set_true_counter]),
                y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fp += tp_and_fp_batch
            batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
            test_set_true_counter += 1
            print('Precistion: {}'.format(batchprecision))
        total_set_precision = tf.divide(total_tp, total_tp_and_fp)
        print('-' * 20 + 'TOTAL PRECISION' + '-' * 20)
        print('Total Precistion on own test set is {}'.format(total_set_precision))
        return total_set_precision


def get_total_recall(dataset, network, val_batch_size, image_size, is_validation_set, own_test_set_true, image_paths):
    """
   Computes the total recall of the network on a given dataset.
   :param dataset: the dataset to evaluate on.
   :param network: the network to evaluate.
   :param val_batch_size: the size of the validation batch.
   :param image_size: the size of the image.
   :param is_validation_set: if the network should be evaluated on the validation set or not.
   :param own_test_set_true: if the network should be evaluated on a custom test set.
   :param image_paths: path of the images for the evaluation (only needed if the network is evaluated on a custom set).
   """
    total_tp = tf.zeros(1, tf.int64)
    total_tp_and_fn = tf.zeros(1, tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    if is_validation_set:
        print()
        for batch in range(num_val_batches):
            x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
            y_pred_logits = network(x, is_training=False)
            y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch, _ = get_confusion_matrix_batch(y_true_labels, y_pred_labels, dataset.num_classes,
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
            print('-' * 20 + image_path + '-' * 20)
            image = read_image(image_path, image_size)
            x = tf.expand_dims(image, axis=0)
            y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
            y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)
            tp_batch, tp_and_fn_batch, _ = get_confusion_matrix_batch(
                np.array(own_test_set_true[test_set_true_counter]),
                y_pred_labels, dataset.num_classes, is_validation_set)
            total_tp += tp_batch
            total_tp_and_fn += tp_and_fn_batch
            batchrecall = tf.divide(tp_batch, tp_and_fn_batch)
            test_set_true_counter += 1
            print('Recall: {}'.format(batchrecall))
        total_set_recall = tf.divide(total_tp, total_tp_and_fn)
        print('-' * 20 + 'TOTAL RECALL' + '-' * 20)
        print('Total Recall on own test set is {}'.format(total_set_recall))
        return total_set_recall


def get_confusion_matrix_batch(y_true, y_pred, num_classes, is_validation_set):
    """
    Computes the confusion matrix for a given batch, i.e. TP, TP+FN, TP+FP.
    Used to compute Precision and Recall.
    :param y_true: the true labels.
    :param y_pred: the predicted labels.
    :param num_classes: number of classes in the dataset.
    :param is_validation_set: if the batch is a validation batch or not.
    """
    tp_batch, tp_and_fn_batch, tp_and_fp_batch = [], [], []
    tp, fn, fp = 0, 0, 0
    if is_validation_set:
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
            if y_true_bool_array[i] == 1 and y_pred_bool_array[i] == 1:
                tp += 1
            if y_pred_bool_array[i] < y_true_bool_array[i]:
                fn += 1
            if y_pred_bool_array[i] > y_true_bool_array[i]:
                fp += 1
        print("y_true: {}".format(y_true_bool_array))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}, False Positives: {}".format(tp, fn, fp))
    else:
        y_pred_labels_list = []
        for class_label in range(num_classes - 1):
            pred_equal_class = tf.cast(tf.equal(y_pred, class_label), tf.int32)
            y_pred_labels_list.append(tf.reduce_sum(pred_equal_class))

        y_pred_array = np.array(y_pred_labels_list)
        y_pred_bool_array = np.where(y_pred_array > 0, 1, 0)

        for i in range(len(y_pred_labels_list)):
            if y_true[i] == 1 and y_pred_bool_array[i] == 1:
                tp += 1
            if y_pred_bool_array[i] < y_true[i]:
                fn += 1
            if y_pred_bool_array[i] > y_true[i]:
                fp += 1
        print("y_true: {}".format(y_true))
        print("y_pred: {}".format(y_pred_bool_array))
        print("True Positives: {}, False Negatives: {}, False Positives: {}".format(tp, fn, fp))
    tp_batch.append(tp)
    tp_and_fn_batch.append(tp + fn)
    tp_and_fp_batch.append(tp + fp)
    return tp_batch, tp_and_fn_batch, tp_and_fp_batch
