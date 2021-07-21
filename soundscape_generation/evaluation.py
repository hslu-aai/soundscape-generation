import argparse
import glob
import os
import tensorflow as tf
from soundscape_generation.models.ERFNet import ERFNet
from soundscape_generation.dataset.cityscapes import CityscapesDataset
from soundscape_generation.eval.evaluation import evaluate, get_total_recall, get_total_percision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_MODEL = "experiments/Cityscapes/ERFNet-Pretrained/pretrained.h5"


def main(args):
    img_h, img_w = args.img_height, args.img_width
    val_batch_size = args.val_batch_size
    is_validation_set = args.is_validation_set


    image_paths = sorted(glob.glob(os.path.join(os.getcwd(), args.test_images, '*[!_pred].{}'.format(args.test_images_type))))  # Specify Image file type

    own_test_set_true = [
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
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
    if eval(args.weights) is None:
        args.weights = DEFAULT_MODEL
    weights_path = os.path.join(os.getcwd(), args.weights)
    network.load_weights(weights_path)
    print('Weights from {} loaded correctly.'.format(weights_path))

    print('*'*20 + 'IOU' + '*'*20)
    iou_per_class, iou_mean = evaluate(dataset, network, val_batch_size, (img_h, img_w))
    print("iou_per_class: {}\niou_mean: {}".format(iou_per_class, iou_mean))

    print('*'*20 + 'PRECISION' + '*'*20)
    get_total_percision(dataset, network, val_batch_size, (img_h, img_w), is_validation_set, own_test_set_true, image_paths)

    print('*'*20 + 'RECALL' + '*'*20)
    get_total_recall(dataset, network, val_batch_size, (img_h, img_w), is_validation_set, own_test_set_true, image_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_validation_set', type=bool, default=False, help='Evaluation on validation or own dataset')
    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--weights', type=str, default="None", help='Relative path of network weights')
    parser.add_argument('--test_images', type=str, default="data/test_images/", help='Relative path of the test images')
    parser.add_argument('--test_images_type', type=str, default="jpg", help='Test image types')
    args = parser.parse_args()

    main(args)
