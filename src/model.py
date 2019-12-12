import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                                    Dropout, Flatten, Input, PReLU,
                                    Reshape, UpSampling2D, ZeroPadding1D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class MultiGan:
    def __init__(self, params):
        self.num_noises, self.num_objs, self.num_labels, self.num_rows, self.num_cols = params
        self.init_object_generator()
        self.init_object_discriminator()
        self.init_gan_model()

    def init_object_generator(self):
        # --- Local generator ---
        g_labels = Input(shape=(self.num_labels, ), dtype=np.float32)
        g_noises = Input(shape=(self.num_noises, ), dtype=np.float32)

        g = Dense(7 * 7 * 256, kernel_initializer='glorot_normal')(concatenate([g_labels, g_noises], axis=1))
        g = BatchNormalization()(g)
        g = PReLU()(g)
        g = Reshape([7, 7, 256])(g)
        g = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = UpSampling2D(size=(2, 2))(g)
        g = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = UpSampling2D(size=(2, 2))(g)
        g = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer='glorot_uniform')(g)
        g = Activation('sigmoid')(g)

        self.obj_generator = Model([g_labels, g_noises], g)
        self.obj_generator_optimizer = Adam()
        self.obj_generator.compile(loss='binary_crossentropy', optimizer=self.obj_generator_optimizer)

    def init_object_discriminator(self):
        # --- Local discriminator ---
        d_images = Input(shape=(28, 28, 1, ), dtype=np.float32)
        d_labels = Input(shape=(self.num_labels, ), dtype=np.float32)

        d = Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2))(d_images)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2))(d)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Conv2D(128, kernel_size=(3, 3), padding='same', strides=(2, 2))(d)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Flatten()(d)
        d = Dense(128)(concatenate([d, d_labels], axis=1))
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)

        self.obj_discriminator = Model([d_images, d_labels], d)
        self.obj_discriminator_optimizer = Adam()
        self.obj_discriminator.compile(loss='categorical_crossentropy', optimizer=self.obj_discriminator_optimizer)

    def init_gan_model(self):
        # --- Global generator ---
        g_labels = Input(shape=(self.num_objs, self.num_labels, ), dtype=np.float32)
        g_bboxes = Input(shape=(self.num_objs, 4, ), dtype=np.int32)
        g_noises = Input(shape=(self.num_noises, ), dtype=np.float32)
        g_render = Input(shape=(self.num_rows, self.num_cols, 1, ), dtype=np.float32)
        chx, chy = 32, 32
        g = Dense(chy * chx * 256, kernel_initializer='glorot_normal')(g_noises)
        g = BatchNormalization()(g)
        g = PReLU()(g)
        g = Reshape([chy, chx, 256])(g)
        g = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = UpSampling2D(size=(2, 2))(g)
        g = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = UpSampling2D(size=(2, 2))(g)
        g = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(g)
        g = BatchNormalization(axis=3)(g)
        g = PReLU()(g)
        g = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer='glorot_uniform')(g)
        g = Activation('sigmoid')(g)
        # Add renders layer.
        # g = concatenate([g, g_render], axis=3)
        for i in range(self.num_objs):
            g = self.generate_object(g, g_labels[:, i, :], g_bboxes[:, i, :], g_noises)

        generator = Model([g_labels, g_bboxes, g_noises, g_render], g)
        g_opt = Adam()
        generator.compile(loss='binary_crossentropy', optimizer=g_opt)

        # --- Global discriminator ---
        d_images = Input(shape=(self.num_rows, self.num_cols, 1, ), dtype=np.float32)
        d_labels = Input(shape=(self.num_objs, self.num_labels, ), dtype=np.float32)
        d_bboxes = Input(shape=(self.num_objs, 4, ), dtype=np.int32)

        d = Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2))(d_images)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2))(d)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Conv2D(128, kernel_size=(3, 3), padding='same', strides=(2, 2))(d)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        d = Flatten()(d)
        d = Dense(128)(d)
        d = PReLU()(d)
        d = Dropout(rate=0.2)(d)
        for i in range(self.num_objs):
            d = self.discriminate_object(d, i, d_images, d_labels[:, i, :], d_bboxes[:, i, :])
        d = Dense(2, activation='softmax')(d)

        discriminator = Model([d_images, d_labels, d_bboxes], d)
        d_opt = Adam()
        discriminator.compile(loss='categorical_crossentropy', optimizer=d_opt)

        # --- GAN ---
        gan_labels = Input(shape=(self.num_objs, self.num_labels, ), dtype=np.float32)
        gan_bboxes = Input(shape=(self.num_objs, 4, ), dtype=np.int32)
        gan_noises = Input(shape=(self.num_noises, ), dtype=np.float32)
        gan_render = Input(shape=(self.num_rows, self.num_cols, 1, ), dtype=np.float32)

        gen_output = generator([gan_labels, gan_bboxes, gan_noises, gan_render])
        gan_output = discriminator([gen_output, gan_labels, gan_bboxes])

        gan = Model([gan_labels, gan_bboxes, gan_noises, gan_render], gan_output)
        gan.compile(loss='categorical_crossentropy', optimizer=g_opt)

        self.generator_optimizer = g_opt
        self.discriminator_optimizer = d_opt
        self.generator, self.discriminator, self.gan = generator, discriminator, gan

    def generate_object(self, g, labels, bboxes, noises):
        bbox = tf.dtypes.cast(bboxes[0], dtype='int32')
        top, left, bottom, right = bbox[0], bbox[1], bbox[2], bbox[3]
        shape = tf.shape(g)
        shape_y, shape_x = shape[1], shape[2]
        right_pad = shape_x - right
        bot_pad = shape_y - bottom

        # Generate object and pad it to image size with regards to its bounding box.
        obj = self.obj_generator([labels, noises])
        obj = tf.pad(obj, [[0, 0], [top, bot_pad], [left, right_pad], [0, 0]], 'CONSTANT', constant_values=1)

        # Combine object with image.
        return tf.minimum(g, obj)

    def discriminate_object(self, d, i, images, labels, bboxes):
        bbox = tf.dtypes.cast(bboxes[0], dtype='int32')
        top, left, bottom, right = bbox[0], bbox[1], bbox[2], bbox[3]

        # Discriminate object.
        obj = self.obj_discriminator([images[:, top:bottom, left:right], labels])

        # Concatenate with result.
        return tf.concat([d, obj], axis=1)
