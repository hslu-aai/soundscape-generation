from abc import ABC

import tensorflow as tf


class DownsamplerBlock(tf.keras.layers.Layer):
    """
    Implementation of the down-sampling Block of the ERFNet.
    """

    def __init__(self, ch_in, ch_out):
        """
        Initializes the layer.
        :param ch_in: number of input channels.
        :param ch_out: number of output channels.
        """
        super(DownsamplerBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(ch_out - ch_in, kernel_size=(3, 3), strides=2, padding='same')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, is_training=True):
        """
        The forward pass through the layer.

        :param x: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        out1 = self.conv(x)
        out2 = self.pool(x)
        out = tf.keras.layers.Concatenate(axis=-1)([out1, out2])
        out = self.bn(out, training=is_training)
        return out


class NonBottleNeck1D(tf.keras.layers.Layer):
    """
    Implementation of the non-bottleneck 1D Block of the ERFNet.
    """

    def __init__(self, ch_out, dropout_rate, dilation_rate):
        """
        Initializes the layer.
        :param ch_out: number of output channels.
        :param dropout_rate: the rate of neurons to randomly drop.
        :param dilation_rate: the dilation rate.
        """
        super(NonBottleNeck1D, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same',
                                            dilation_rate=(dilation_rate, 1))
        self.conv4 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same',
                                            dilation_rate=(1, dilation_rate))
        self.bn2 = tf.keras.layers.BatchNormalization()
        if self.dropout_rate != 0:
            self.drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inp, is_training=True):
        """
        The forward pass through the layer.

        :param inp: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        x = inp
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = self.bn2(x, training=is_training)

        if self.dropout_rate != 0:
            x = self.drop(x, training=is_training)

        x = tf.keras.layers.Add()([x, inp])
        x = tf.nn.relu(x)
        return x


class UpsamplerBlock(tf.keras.layers.Layer):
    """
    Implementation of the up-sampling Block of the ERFNet.
    """

    def __init__(self, ch_out):
        """
        Initializes the layer.
        :param ch_out: number of output channels.
        """
        super(UpsamplerBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(ch_out, kernel_size=(3, 3), strides=2, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, is_training=True):
        """
        The forward pass through the layer.

        :param x: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)
        return x


class Encoder(tf.keras.Model, ABC):
    """
    Implementation of the Encoder part of the ERFNet.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        super(Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(ch_in=3, ch_out=16)

        self.blocks = []
        self.blocks.append(DownsamplerBlock(ch_in=16, ch_out=64))

        for _ in range(5):
            self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0.03, dilation_rate=1))

        self.blocks.append(DownsamplerBlock(ch_in=64, ch_out=128))

        for _ in range(2):
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=2))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=4))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=8))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=16))

    def call(self, x, is_training=True, **kwargs):
        """
        The forward pass through the model.

        :param x: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        x = self.initial_block(x, training=is_training)
        for block in self.blocks:
            x = block(x, training=is_training)
        return x


class Decoder(tf.keras.Model, ABC):
    """
    Implementation of the Decoder part of the ERFNet.
    """

    def __init__(self, num_classes):
        """
        Initializes the model.
        :param num_classes: number of output classes.
        """
        super(Decoder, self).__init__()
        self.blocks = []
        self.blocks.append(UpsamplerBlock(ch_out=64))
        self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0, dilation_rate=1))
        self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0, dilation_rate=1))
        self.blocks.append(UpsamplerBlock(ch_out=16))
        self.blocks.append(NonBottleNeck1D(ch_out=16, dropout_rate=0, dilation_rate=1))
        self.blocks.append(NonBottleNeck1D(ch_out=16, dropout_rate=0, dilation_rate=1))
        self.output_conv = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=2, padding='same')

    def call(self, x, is_training=True, **kwargs):
        """
        The forward pass through the model.

        :param x: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        for block in self.blocks:
            x = block(x, training=is_training)
        x = self.output_conv(x)
        return x


class ERFNet(tf.keras.Model, ABC):
    """
    Implementation of the ERFNet.
    """

    def __init__(self, num_classes):
        """
        Initializes the model.
        :param num_classes: number of output classes.
        """
        super(ERFNet, self).__init__()
        self.model_name = 'ERFNet'
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def call(self, x, is_training=True, **kwargs):
        """
        The forward pass through the model.

        :param x: the input that will be passed through the model.
        :param is_training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        out = self.encoder(x, training=is_training)
        out = self.decoder(out, training=is_training)
        return out
