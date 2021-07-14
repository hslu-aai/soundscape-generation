import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Constants
PATHTOIMAGEFOLDER = "C:/Users/Majke/OneDrive/HSLU/6.Semester/BA/Testbild/"
PATHTOFOREGROUNDFOLDER = "C:/Soundbank/foreground"
PATHTOBACKGROUNDFOLDER = "C:/Soundbank/background"

class SimpleObjectDection():
    def __init__(self, annotationsJsonFile : str = "bochum_000000_000313_gtFine_polygons.json", annotatedImageFileName : str = "bochum_000000_000313_gtFine_color.png", originalImageFileName : str = "bochum_000000_000313_leftImg8bit.png"):
        with open(PATHTOIMAGEFOLDER + annotationsJsonFile, "r") as read_file:
            self.data = json.load(read_file)['objects']
            self.annotatedImage = annotatedImageFileName
            self.originalImage = originalImageFileName

    def get_labels_from_json(self) -> list:
        labelsWithDuplicates = []
        for i in range(len(self.data)):
            labelsWithDuplicates.append(self.data[i]['label'])
        labels = list(dict.fromkeys(labelsWithDuplicates))
        return labels
    
    def plot_image(self) -> None:
        imageAnnotation = mpimg.imread(PATHTOIMAGEFOLDER + self.annotatedImage)
        cityImage = mpimg.imread(PATHTOIMAGEFOLDER + self.originalImage)
        plt.subplot(121)
        plt.title("Original Image")
        plt.imshow(cityImage)
        plt.subplot(122)
        plt.title("Annotated Image")
        plt.imshow(imageAnnotation)
        plt.show()

    def get_foreground_sounds(self):
        allForegroundSoundsList = os.listdir(PATHTOFOREGROUNDFOLDER)
        detectedSounds = self.get_labels_from_json() 
        foregroundSoundList = []
        for i in range(len(detectedSounds)):
            if detectedSounds[i] in allForegroundSoundsList:
                foregroundSoundList.append(detectedSounds[i])
        return foregroundSoundList

    def get_background_sounds(self):
        allBackgroundSounds = os.listdir(PATHTOBACKGROUNDFOLDER)
        detectedSounds = self.get_labels_from_json()
        backgroundSoundList = []
        for i in range(len(detectedSounds)):
            if detectedSounds[i] in allBackgroundSounds:
                backgroundSoundList.append(detectedSounds[i])
        return backgroundSoundList

if __name__ == "__main__":
    city_dataset = SimpleObjectDection()
    city_dataset.plot_image()
    print(city_dataset.get_labels_from_json())
    print(city_dataset.get_foreground_sounds())
    print(city_dataset.get_background_sounds())
else:
    city_dataset = SimpleObjectDection()
    city_dataset.plot_image()    
    