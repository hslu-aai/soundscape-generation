# Extended Prototype
## Soundgeneration module
### Introduction
The sound generation module was developed with Scaper. Given a collection of isolated sound events, Scaper acts as a high-level sequencer that can generate multiple soundscapes from a single, probabilistically defined, “specification”. 

### Installation
Follow the instructions give in the following link:
* [Scaper installation](https://scaper.readthedocs.io/en/latest/installation.html)

### Generate soundscapes
Run the file soundGeneration.py to generate soundscapes of every image in the test_images/ directory (note: results are saved in the soundscapes/ directory). Make sure that you specify the file type of the image in the image path variable of predict.py.

#### References
* [J. Salamon, D. MacConnell, M. Cartwright, P. Li and J. P. Bello, "Scaper: A library for soundscape synthesis and augmentation," 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2017, pp. 344-348, doi: 10.1109/WASPAA.2017.8170052.](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_scaper_waspaa_2017.pdf)

## Object Detection module

### Credits

Thanks to garder14 for making this pre-trained model available.

### ERFNet - TensorFlow 2

This is an unofficial implementation of [ERFNet](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) for semantic segmentation on the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

#### Results

![](assets/test1.png)
![](assets/test2.png)
![](assets/test3.png)

The above predictions are produced by a network trained for 67 epochs that achieves a mean class IoU score of 0.7084 on the validation set. To get this model, I completed 70 epochs (almost 10 hours on a single Tesla P100 GPU) and selected the checkpoint with maximum validation score. The progression of this metric is shown below.

<img src="assets/iou_plot.png" width="65%">

The inference time on a Tesla P100 GPU is around 0.2 seconds per image.

#### Separate Software installation of the pre-trained model

Clone this repository:

```bash
git clone https://github.com/garder14/erfnet-tensorflow2.git
cd erfnet-tensorflow2/
```

Dependencies of the pre-trained model:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=11.0.221
conda activate tf-gpu
pip install tensorflow_addons==0.13.0 Pillow==7.1.2
```

#### Training

Before training the network, you need to download the Cityscapes dataset. For this purpose, create an account in [www.cityscapes-dataset.com](https://www.cityscapes-dataset.com/), and run the following command (indicating your username and password):

```bash
bash download_data.sh username password
```

To train the network, run this command:

```bash
python train.py --num_epochs 70 --batch_size 8 --evaluate_every 1 --save_weights_every 1
```

By default, training resumes from the latest saved checkpoint. If the checkpoints/ directory is missing, the training starts from zero.

#### Inference

Run the following command to predict the semantic segmentation of every image in the test_images/ directory (note: results are saved in the test_segmentations/ directory)

```bash
python predict.py
```
Make sure that you specify the file type of the image in the image path variable in predict.py.

#### References

* [E. Romera et al., "ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", 2017](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

* [Official PyTorch implementation of ERFNet](https://github.com/Eromera/erfnet_pytorch)
