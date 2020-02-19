from keras.models import Model, Input
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Concatenate, UpSampling2D


class SegNet:
    def __init__(self, n_filters, input_dim_x, input_dim_y, num_channels):
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.n_filters = n_filters
        self.num_channels = num_channels

    def get_SegNet(self):
        convnet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        encoder_conv1 = Conv2D(self.n_filters, kernel_size=9, activation='relu', padding='same')(convnet_input)
        pool1 = MaxPooling2D(pool_size=(2, 2))(encoder_conv1)
        encoder_conv2 = Conv2D(self.n_filters, kernel_size=5, activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(encoder_conv2)
        encoder_conv3 = Conv2D(self.n_filters * 2, kernel_size=5, activation='relu', padding='same')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(encoder_conv3)
        encoder_conv4 = Conv2D(self.n_filters * 2, kernel_size=5, activation='relu', padding='same')(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(encoder_conv4)

        conv5 = Conv2D(self.n_filters, kernel_size=5, activation='relu', padding='same')(pool4)

        decoder_conv6 = Conv2D(self.n_filters, kernel_size=7, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        decoder_conv7 = Conv2D(self.n_filters, kernel_size=5, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(decoder_conv6))
        decoder_conv8 = Conv2D(self.n_filters, kernel_size=5, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(decoder_conv7))
        #decoder_conv9 = Conv2D(self.n_filters, kernel_size=5, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(decoder_conv8))
        decoder_conv9 = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(UpSampling2D(size=(2, 2))(decoder_conv8))

        return Model(outputs=decoder_conv9, inputs=convnet_input), 'SegNet'