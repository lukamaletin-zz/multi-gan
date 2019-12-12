import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from model import MultiGan


MODEL_PATH = r'../out/gan_2.h5'


class Tester:
    def __init__(self):
        self.num_noises = 100
        self.num_samples = 10000
        self.num_objs = 1
        self.num_labels = 10
        sample_rows = 128
        sample_cols = 128

        params = (self.num_noises, self.num_objs, self.num_labels, sample_rows, sample_cols)
        self.model = MultiGan(params)
        self.model.gan.load_weights(MODEL_PATH)

        self.obj_dump_input = self.init_obj_dump_input()
        self.classifier = self.init_classifier()

    def init_obj_dump_input(self):
        (_, _), (_, y_test) = mnist.load_data()
        labels_batch = to_categorical(y_test, self.num_labels)
        noises_batch = np.random.uniform(0, 1, size=(self.num_samples, self.num_noises))

        return [labels_batch, noises_batch]

    def init_classifier(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = 1 - x_train
        x_test = 1 - x_test
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        classifier = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        classifier.fit(x_train, y_train, epochs=5)
        classifier.evaluate(x_test,  y_test, verbose=2)

        return classifier

    def test(self):
        obj_dump_output = self.model.obj_generator.predict(self.obj_dump_input)
        images = obj_dump_output
        labels = np.where(self.obj_dump_input[0]==1)[1]
        self.classifier.evaluate(images, labels, verbose=2)


if __name__ == '__main__':
    tester = Tester()
    tester.test()
