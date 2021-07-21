# Soundscape Generation

Generate soundscapes from images.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [References](#references)

## Installation

### Scaper Installation

The sound generation module was developed using Scaper. Given a collection of isolated sound events, Scaper acts as a
high-level sequencer that can generate multiple soundscapes from a single probabilistically defined specification.

Follow the instructions give in the following link:

* [Scaper installation](https://scaper.readthedocs.io/en/latest/installation.html)

### Download Dependencies

```bash
pip install -r requirements.txt
```

### Download Cityscapes Dataset

To download the dataset, a cityscapes account is required for the authentification. Such an account can be created
on [www.cityscapes-dataset.com](https://www.cityscapes-dataset.com/). After the registration, run the `download_data.sh`
script. During the download, it will ask you to provide your email and password for authentification.

```bash
./scripts/download_data.sh
```

## Usage

For the object detection module a
pre-trained [ERFNet](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) is used, which is then
finetuned on the Cityscapes dataset.

### Train Object Segmentation Network

To train the network, run the follwing command. The hyperparameters epoch and batch size can be configured in the `docker-compose.yml` file. To load a pre-trained model specify its path in the `MODEL_TO_LOAD` variable, if the variable is `None` the model is trained from scratch.

```bash
docker-compose up train_object_detection
```

### Test the Segmentation Network

Run the following command to predict the semantic segmentation of every image in the `--test_images` directory (note:
predictions are saved with the same name and a `_pred.jpg` suffix). Ensure that you specify the correct image's file type in `--test_images_type`.

```bash
docker-compose up predict_object_detection
```

### Evaluate the Segmentation Network
To evaluate the segmentation network run the command below.

```bash
docker-compose up evaluation
```

### Generate soundscapes

To generate soundscapes of every image in the `--test_images` directory run the following command. The generated audios will be saved in `data/soundscapes`. Ensure that you specify the correct image's file type in `--test_images_type`.

```bash
docker-compose up sound_generation
```

## Results

### Object Detection

![](assets/test1.png)
![](assets/test2.png)
![](assets/test3.png)

The above predictions are produced by a network trained for 67 epochs that achieves a mean class IoU score of 0.7084 on
the validation set. The inference time on a Tesla P100 GPU is around 0.2 seconds per image. The model was trained for 70 epochs on a single Tesla P100. After the training, the checkpoint
that yielded to highest validation IoU score was selected. The progression of the IoU metric is shown below.

![](assets/iou_plot.png)

## References

* [J. Salamon, D. MacConnell, M. Cartwright, P. Li and J. P. Bello, "Scaper: A library for soundscape synthesis and augmentation," 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2017, pp. 344-348, DOI: 10.1109/WASPAA.2017.8170052.](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_scaper_waspaa_2017.pdf)
* [E. Romera et al., "ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", 2017](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
* [Official PyTorch implementation of ERFNet](https://github.com/Eromera/erfnet_pytorch)
