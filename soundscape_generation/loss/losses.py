import tensorflow as tf


def weighted_cross_entropy_loss(y_true_labels, y_pred_logits, class_weights):
    """
    Computes the weighted cross entropy loss of a given batch.
    The class weights should be pre-computed for the entire dataset.
    Note: The predicted labels need to be one-hot encoded.
    :param y_true_labels: true labels of the batch (batch_size, img_h, img_w).
    :param y_pred_logits: predicted labels of the batch (batch_size, img_h, img_w, num_classes).
    :param class_weights: class weights of the dataset (e.g. fraction of each label in the dataset.
    :return: weighted cross entropy loss.
    """
    # loss gets computed pixel-wise
    # output shape: (batch_size, img_h, img_w)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_labels,
                                                            logits=y_pred_logits)
    # multiply the loss by the class weights to get the weighted loss
    # ouput shape: (batch_size, img_h, img_w)
    weights = tf.gather(class_weights, y_true_labels)
    losses = tf.multiply(losses, weights)

    return tf.reduce_mean(losses)
