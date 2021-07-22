import numpy as np
import os
import scaper

from soundscape_generation.utils.utils import create_folder


class SoundGenerator:
    """ Constansts for Scaper """
    # OUTPUT FOLDER
    OUTFOLDER = "data/generated_soundscapes/"

    # SCAPER SETTINGS
    FG_FOLDER = "data/soundbank/foreground"
    BG_FOLDER = "data/soundbank/background"
    REF_DB = -3  # Difference between background and foreground DB
    DURATION = 30.0

    MIN_EVENTS = 4
    MAX_EVENTS = 7  # 10 Objects with sound - 3 Background = 7

    EVENT_TIME_DIST = 'normal'
    EVENT_TIME_MEAN = 20
    EVENT_TIME_STD = 9

    SOURCE_TIME_DIST = 'const'
    SOURCE_TIME = 0.0

    EVENT_DURATION_DIST = 'uniform'
    EVENT_DURATION_MIN = 12
    EVENT_DURATION_MAX = 16

    SNR_DIST = 'uniform'  # the signal-to-noise ratio (in LUFS) compared to the background (DB Difference).
    SNR_MIN = 3
    SNR_MAX = 5

    PITCH_DIST = 'uniform'
    PITCH_MIN = -0.2
    PITCH_MAX = 0.2

    TIME_STRETCH_DIST = 'uniform'
    TIME_STRETCH_MIN = 0.5
    TIME_STRETCH_MAX = 1.0

    SEED = 123

    def __init__(self, foreground_sounds, background_sounds, image_names):
        """
        Initialisation of Scaper and Object-Detection Container.
        :param foreground_sounds: list of foreground sounds.
        :param background_sounds: list of background sounds.
        :param image_names: list of images from which to generate soundscapes.
        """
        self.sc = scaper.Scaper(self.DURATION, self.FG_FOLDER, self.BG_FOLDER, random_state=self.SEED)
        self.sc.protected_labels = []
        self.sc.ref_db = self.REF_DB
        self.detected_foreground_sounds = foreground_sounds
        self.detected_background_sounds = background_sounds
        self.image_names = image_names

        # create the output folder
        create_folder(os.path.join(os.getcwd(), self.OUTFOLDER))

    def generate_sound(self, n_soundscapes):
        """
        Generate $n$ number of soundscapes from the given `image_names` (i.e. segmented images)
        using truncated normal distribution of start times.
        :param n_soundscapes: number of soundscapes that should be generated from each image.
        :return: None.
        """
        print('*' * 20 + 'START GENERATION' + '*' * 20)
        for i in range(len(self.image_names)):
            image_name = self.image_names[i]
            image_name = image_name.split('.')[0].strip()
            print('-' * 20 + 'GENERATION: {}'.format(image_name) + '-' * 20)

            fg_sound = self.detected_foreground_sounds[i]
            bg_sound = self.detected_background_sounds[i]
            all_foreground_sounds_list = os.listdir(self.FG_FOLDER)
            all_background_sounds_list = os.listdir(self.BG_FOLDER)
            final_fg_sound = [x for x in fg_sound if x in all_foreground_sounds_list]
            final_bg_sound = [x for x in bg_sound if x in all_background_sounds_list]

            for n in range(len(n_soundscapes)):
                # reset the event specifications for foreground and background at the
                # beginning of each loop to clear all previously added events
                self.sc.reset_bg_event_spec()
                self.sc.reset_fg_event_spec()

                # add background
                self.sc.add_background(label=('choose', final_bg_sound),
                                       source_file=('choose', []),
                                       source_time=('normal', 20, 8))

                # add random number of foreground events
                n_events = np.random.randint(self.MIN_EVENTS, self.MAX_EVENTS + 1)
                for _ in range(n_events):
                    self.sc.add_event(label=('choose', final_fg_sound),
                                      source_file=('choose', []),
                                      source_time=(self.SOURCE_TIME_DIST, self.SOURCE_TIME),
                                      event_time=(self.EVENT_TIME_DIST, self.EVENT_TIME_MEAN, self.EVENT_TIME_STD),
                                      event_duration=(
                                      self.EVENT_DURATION_DIST, self.EVENT_DURATION_MIN, self.EVENT_DURATION_MAX),
                                      snr=(self.SNR_DIST, self.SNR_MIN, self.SNR_MAX),
                                      pitch_shift=None,
                                      time_stretch=None)

                # define the output files
                audio_file = os.path.join(self.OUTFOLDER, "{}_soundscape_number_{:d}.wav".format(image_name, n + 1))
                txt_file = os.path.join(self.OUTFOLDER, "{}_soundscape_number_{:d}.txt".format(image_name, n + 1))

                # generate the soundscape
                self.sc.generate(audio_file,
                                 allow_repeated_label=True,
                                 allow_repeated_source=True,
                                 reverb=0.1,
                                 disable_sox_warnings=True,
                                 no_audio=False,
                                 txt_path=txt_file,
                                 peak_normalization=True,
                                 disable_instantiation_warnings=True)

                print('Generated soundscape: {}'.format(audio_file))
