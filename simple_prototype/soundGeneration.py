
import os
from simpleObjectDetection import SimpleObjectDection
import scaper
import numpy as np


""" Constansts for Scaper """
# OUTPUT FOLDER
OUTFOLDER = "C:/Soundscape"
# SCAPER SETTINGS
FG_FOLDER = "C:/Soundbank/foreground"
BG_FOLDER = "C:/Soundbank/background"
N_SOUNDSCAPES = 2
REF_DB = -50
DURATION = 30.0

MIN_EVENTS = 3
MAX_EVENTS = 9

EVENT_TIME_DIST = 'truncnorm'
EVENT_TIME_MEAN = 5.0
EVENT_TIME_STD = 2.0
EVENT_TIME_MIN = 0.0
EVENT_TIME_MAX = 10.0

SOURCE_TIME_DIST = 'const'
SOURCE_TIME = 0.0

EVENT_DURATION_DIST = 'uniform'
EVENT_DURATION_MIN = 7
EVENT_DURATION_MAX = 12

SNR_DIST = 'uniform'
SNR_MIN = 6
SNR_MAX = 30

PITCH_DIST = 'uniform'
PITCH_MIN = -1.0
PITCH_MAX = 1.0

TIME_STRETCH_DIST = 'uniform'
TIME_STRETCH_MIN = 0.8
TIME_STRETCH_MAX = 1.2
# generate a random seed for this Scaper object
SEED = 123

class SoundGenerator():

    def __init__(self):
        # Initialisation of Scaper and Object-Detection Container
        self.sc = scaper.Scaper(DURATION, FG_FOLDER, BG_FOLDER, random_state=SEED)
        self.sc.protected_labels = []
        self.sc.ref_db = REF_DB

        self.detectedObjectsContainer = SimpleObjectDection()
        self.detectedForegroundSounds = self.detectedObjectsContainer.get_foreground_sounds()
        self.detectedBackgroundSounds = self.detectedObjectsContainer.get_background_sounds()

    # Generate 2 soundscapes using a truncated normal distribution of start times
    def generateSound(self):
        for n in range(N_SOUNDSCAPES):

            print('Generating soundscape: {:d}/{:d}'.format(n+1, N_SOUNDSCAPES))

            # reset the event specifications for foreground and background at the
            # beginning of each loop to clear all previously added events
            self.sc.reset_bg_event_spec()
            self.sc.reset_fg_event_spec()

            # add background
            self.sc.add_background(label=('choose', self.detectedBackgroundSounds),
                            source_file=('choose', []),
                            source_time=('const', 0))

            # add random number of foreground events
            n_events = np.random.randint(MIN_EVENTS, MAX_EVENTS+1)
            for _ in range(n_events):
                self.sc.add_event(label=('choose', self.detectedForegroundSounds),
                            source_file=('choose', []),
                            source_time=(SOURCE_TIME_DIST, SOURCE_TIME),
                            event_time=(EVENT_TIME_DIST, EVENT_TIME_MEAN, EVENT_TIME_STD, EVENT_TIME_MIN, EVENT_TIME_MAX),
                            event_duration=(EVENT_DURATION_DIST, EVENT_DURATION_MIN, EVENT_DURATION_MAX),
                            snr=(SNR_DIST, SNR_MIN, SNR_MAX),
                            pitch_shift=(PITCH_DIST, PITCH_MIN, PITCH_MAX),
                            time_stretch=(TIME_STRETCH_DIST, TIME_STRETCH_MIN, TIME_STRETCH_MAX))

            # generate
            audiofile = os.path.join(OUTFOLDER, "soundscape_unimodal{:d}.wav".format(n))
            jamsfile = os.path.join(OUTFOLDER, "soundscape_unimodal{:d}.jams".format(n))
            txtfile = os.path.join(OUTFOLDER, "soundscape_unimodal{:d}.txt".format(n))

            self.sc.generate(audiofile, jamsfile,
                        allow_repeated_label=True,
                        allow_repeated_source=True,
                        reverb=0.1,
                        disable_sox_warnings=True,
                        no_audio=False,
                        txt_path=txtfile)
                        
if __name__ == "__main__":
    soundGenerator = SoundGenerator()
    print(soundGenerator.detectedBackgroundSounds)
    print(soundGenerator.detectedForegroundSounds)
    soundGenerator.generateSound()