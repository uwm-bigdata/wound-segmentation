from keras.models import Model, Input
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Concatenate, UpSampling2D


class Unet2D:

    def __init__(self, n_filters, input_dim_x, input_dim_y, num_channels):
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.n_filters = n_filters
        self.num_channels = num_channels

    def get_unet_model_5_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))
        
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        up6 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        concat6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        up7 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        
        up8 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        concat8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        
        up9 = Conv2D(self.n_filters*2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        concat9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv9)
        
        return Model(outputs=conv10,  inputs=unet_input), 'unet_model_5_levels'


    def get_unet_model_4_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))
                
        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        
        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        
        up5 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
        concat5 = Concatenate()([drop3, up5])
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat5)
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        
        up6 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        concat6 = Concatenate()([conv2, up6])
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        up7 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv1, up7])
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        conv9 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv7)
        
        return Model(outputs=conv9,  inputs=unet_input), 'unet_model_4_levels'


    def get_unet_model_yuanqing(self):
        # Model inspired by https://github.com/yuanqing811/ISIC2018
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)

        up6 = Conv2D(self.n_filters * 4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        feature4 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv4)
        concat6 = Concatenate()([feature4, up6])
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(self.n_filters * 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        feature3 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv3)
        concat7 = Concatenate()([feature3, up7])
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(self.n_filters * 1, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        feature2 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv2)
        concat8 = Concatenate()([feature2, up8])
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(int(self.n_filters / 2), 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        feature1 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv1)
        concat9 = Concatenate()([feature1, up9])
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(3, kernel_size=3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)

        return Model(outputs=conv10, inputs=unet_input), 'unet_model_yuanqing'
