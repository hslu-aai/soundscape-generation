import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import time
from soundscape_generation.models.ERFNet import ERFNet
from soundscape_generation.dataset.cityscapes import CityscapesDataset
from soundscape_generation.eval.evaluation import compute_iou
from soundscape_generation.loss.losses import weighted_cross_entropy_loss
from soundscape_generation.utils.utils import create_folder_for_experiment, set_gpu_experimental_growth

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):
    set_gpu_experimental_growth()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    img_h, img_w = args.img_height, args.img_width

    # create dataset and model
    dataset = CityscapesDataset(image_size=(img_h, img_w))
    network = ERFNet(dataset.num_classes)

    if eval(args.model_to_load) is None:
        experiment_path = create_folder_for_experiment(network.model_name, dataset.name)
    else:
        experiment_path = os.path.join(os.getcwd(), args.model_to_load)

    # initialize weights of the network
    inp_test = tf.random.normal(shape=(batch_size, img_h, img_w, 3))
    out_test = network(inp_test, is_training=False)
    print('Network created. Output shape: {}.'.format(out_test.shape))

    num_batches_per_epoch = dataset.num_images // batch_size
    print('Update steps per epoch:', num_batches_per_epoch)
    total_update_steps = num_epochs * num_batches_per_epoch
    print('Total update steps:', total_update_steps)

    # define optimizer (with learning rate schedule)
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=8e-4,
                                                                decay_steps=total_update_steps,
                                                                end_learning_rate=0.0,
                                                                power=0.9)
    opt = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    # manage checkpoints (keep track of network weights, optimizer state, and last epoch id)
    ckpt_path = os.path.join(experiment_path, 'checkpoints')
    ckpt = tf.train.Checkpoint(network=network, opt=opt, last_saved_epoch=tf.Variable(-1))
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        initial_epoch = int(ckpt.last_saved_epoch + 1)
        print('Latest checkpoint restored ({}). Training from epoch {}.'.format(ckpt_manager.latest_checkpoint,
                                                                                initial_epoch + 1))
    else:
        initial_epoch = 0
        print('No checkpoint restored. Training from scratch.')

    # training
    start = time.time()
    for epoch in range(initial_epoch, num_epochs):
        dataset.shuffle_training_paths()
        for batch in range(num_batches_per_epoch):
            x, y_true_labels = dataset.get_training_batch(batch, batch_size)
            loss = train_step(network, dataset, opt, x, y_true_labels)

            # print information about the current batch
            if (batch + 1) % args.print_every == 0:
                current_step = int(opt.iterations)
                current_lr = opt.learning_rate(current_step)
                elapsed_time = time.time() - start
                print('[Epoch {}/{}. Batch {}/{}]'.format(epoch + 1, num_epochs, batch + 1, num_batches_per_epoch),
                      end=' ')
                print('Training batch loss: {:.2f}. Elapsed time: {:.1f} s. Schedule: (step {}, lr {:.1e}).'.format(
                    loss, elapsed_time, current_step, current_lr))

        # save training checkpoint after each epoch
        ckpt.last_saved_epoch.assign_add(1)
        ckpt_save_path = ckpt_manager.save()
        print('Checkpoint saved at {}.'.format(ckpt_save_path))

        # save current weights as a h5 file
        if (epoch + 1) % args.save_weights_every == 0:
            h5_save_path = os.path.join(experiment_path, 'saved_weights', 'weights-{}.h5'.format(epoch + 1))
            network.save_weights(h5_save_path)
            print('Weights saved at {}.'.format(h5_save_path))

        # compute the IoU score on validation set
        if (epoch + 1) % args.evaluate_every == 0:
            iou_per_class, iou_mean = compute_iou(dataset, network, val_batch_size, (img_h, img_w))
            print('Mean IoU: {:.4f}'.format(iou_mean))
            with np.printoptions(formatter={'float': '{:.4f}'.format}):
                print('IoU per class: {}'.format(iou_per_class.numpy()))


@tf.function
def train_step(model, dataset, opt, x, y_true_labels):
    print('Tracing training step...')

    # forward pass
    with tf.GradientTape() as tape:
        y_pred_logits = model(x)  # (batch_size, img_h, img_w, num_classes)
        loss = weighted_cross_entropy_loss(y_true_labels, y_pred_logits, dataset.class_weights)

    # backward pass
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs')

    parser.add_argument('--print_every', type=int, default=5, help='Number of batches between batch logs')
    parser.add_argument('--evaluate_every', type=int, default=1, help='Number of epochs between evaluations')
    parser.add_argument('--save_weights_every', type=int, default=1, help='Number of epochs between saves')
    parser.add_argument('--model_to_load', type=str, default=None, help='Path of the pre-trained model that should be loaded')

    args = parser.parse_args()
    main(args)
