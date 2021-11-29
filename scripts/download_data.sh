#!/bin/bash

echo 'Current Path:'
echo $PWD

# data folder
mkdir -p ./data
python3 -m cityscapesscripts.download.downloader -l
python3 -m cityscapesscripts.download.downloader

# segmentations
python3 -m cityscapesscripts.download.downloader -d data/ gtFine_trainvaltest.zip
wait
unzip data/gtFine_trainvaltest.zip -d data/
wait
rm data/gtFine_trainvaltest.zip data/license.txt data/README
wait
mv ./data/gtFine ./data/segmentations

# images
python3 -m cityscapesscripts.download.downloader -d data/ leftImg8bit_trainvaltest.zip
wait
unzip data/leftImg8bit_trainvaltest.zip -d data/
wait
rm data/leftImg8bit_trainvaltest.zip data/license.txt data/README
wait
mv ./data/leftImg8bit ./data/images

echo 'Finished downloading the dataset.'
