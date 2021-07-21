import argparse
import warnings

import soundscape_generation.predict as predict
from soundscape_generation.generation.sound_generation import SoundGenerator

warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--weights', type=str, default="None", help='Relative path of network weights')
    parser.add_argument('--test_images', type=str, default="data/test_images/", help='Relative path of the test images')
    parser.add_argument('--test_images_type', type=str, default="jpg", help='Test image types')
    args = parser.parse_args()

    # predict the test images to get the objects in fore- and background
    foreground_objects, background_objects = predict.main(args)
    image_names = list(foreground_objects.keys())
    foreground_objects_list = list(foreground_objects.values())
    background_objects_list = list(background_objects.values())

    # instantiate the sound generator
    soundscape_generator = SoundGenerator(foreground_sounds=foreground_objects_list,
                                          background_sounds=background_objects_list,
                                          image_names=image_names)
    # generate for each image 3 different soundscapes
    number_soundscapes_per_image = range(0, 3)
    soundscape_generator.generate_sound(number_soundscapes_per_image)
