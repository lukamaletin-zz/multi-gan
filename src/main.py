import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm

import data
import util
from model import MultiGan


DUMP_SIZE = 10
LOSSES = {'g': [], 'd': []}
OUT_PATH = '../out'


class Trainer:
    def __init__(self):
        dataset = data.load_dataset()
        self.images_train, self.labels_train, self.bboxes_train = dataset

        self.num_noises = 100
        self.num_samples = self.images_train.shape[0]
        self.sample_shape = self.images_train.shape[1:]
        self.num_objs = self.labels_train.shape[1]
        self.num_labels = self.labels_train.shape[2]
        sample_rows = self.images_train.shape[1]
        sample_cols = self.images_train.shape[2]

        params = (self.num_noises, self.num_objs, self.num_labels, sample_rows, sample_cols)
        self.model = MultiGan(params)

        self.dump_input = self.init_dump_input()
        self.obj_dump_input = self.init_obj_dump_input()

    def run_training(self, num_epochs=100, start_at=0, batch_size=32):
        for epoch in tqdm(range(start_at, start_at + num_epochs)):
            # Take random batch samples.
            take_idx = np.random.randint(0, self.num_samples, size=batch_size)
            images_batch = self.images_train[take_idx]
            labels_batch = self.labels_train[take_idx]
            bboxes_batch = self.bboxes_train[take_idx]
            # Create random noise batch.
            noises_batch = np.random.uniform(0, 1, size=(batch_size, self.num_noises))
            # Create renders of bounding boxes on blank canvas.
            render_batch = util.render_bboxes(bboxes_batch, labels_batch, shape=self.sample_shape)

            # --- 1. Generate fake images. ---
            g_input = [labels_batch, bboxes_batch, noises_batch, render_batch]
            g_output = self.model.generator.predict(g_input)

            if epoch % 10 == 0:
                dump_output = self.model.generator.predict(self.dump_input)
                util.save_batch(dump_output, epoch, OUT_PATH)
                obj_dump_output = self.model.obj_generator.predict(self.obj_dump_input)
                util.save_batch(obj_dump_output, epoch, OUT_PATH, 'obj')

            # --- 2. Train discriminator on batch. ---
            util.make_trainable(self.model.discriminator, True)
            util.make_trainable(self.model.obj_discriminator, True)
            # Join real and fake images.
            d_images = np.concatenate((images_batch, g_output))
            d_labels = np.concatenate((labels_batch, labels_batch))
            d_bboxes = np.concatenate((bboxes_batch, bboxes_batch))
            d_input = [d_images, d_labels, d_bboxes]
            # Set target classes with smoothing. First half are real, second half are fake.
            targets = np.zeros([2 * batch_size, 2])
            targets[0:batch_size, 0] = np.random.uniform(0.0, 0.3, size=batch_size)
            targets[0:batch_size, 1] = np.random.uniform(0.7, 1.0, size=batch_size)
            targets[batch_size:, 0] = np.random.uniform(0.7, 1.0, size=batch_size)
            targets[batch_size:, 1] = np.random.uniform(0.0, 0.3, size=batch_size)

            d_loss = self.model.discriminator.train_on_batch(d_input, targets)
            LOSSES['d'].append(d_loss)

            # --- 3. Train generator on batch. ---
            util.make_trainable(self.model.discriminator, False)
            util.make_trainable(self.model.obj_discriminator, False)
            noises_batch = np.random.uniform(0, 1, size=(batch_size, self.num_noises))
            gan_input = [labels_batch, bboxes_batch, noises_batch, render_batch]
            # Set target classes. We want generated images to be classified as real.
            targets = np.zeros([batch_size, 2])
            targets[:, 0] = np.random.uniform(0.0, 0.3, size=batch_size)
            targets[:, 1] = np.random.uniform(0.7, 1.0, size=batch_size)

            g_loss = self.model.gan.train_on_batch(gan_input, targets)
            LOSSES['g'].append(g_loss)

    def init_dump_input(self):
        labels_batch = np.zeros((DUMP_SIZE, self.num_objs, self.num_labels), dtype=np.float32)
        # First object:
        if self.num_objs > 0:
            labels_batch[0, 0, 0] = labels_batch[1, 0, 1] = 1
            labels_batch[2, 0, 2] = labels_batch[3, 0, 3] = 1
            labels_batch[4, 0, 4] = labels_batch[5, 0, 5] = 1
            labels_batch[6, 0, 6] = labels_batch[7, 0, 7] = 1
            labels_batch[8, 0, 8] = labels_batch[9, 0, 9] = 1
        # Second object:
        if self.num_objs > 1:
            labels_batch[0, 1, 1] = labels_batch[1, 1, 2] = 1
            labels_batch[2, 1, 3] = labels_batch[3, 1, 4] = 1
            labels_batch[4, 1, 5] = labels_batch[5, 1, 6] = 1
            labels_batch[6, 1, 7] = labels_batch[7, 1, 8] = 1
            labels_batch[8, 1, 9] = labels_batch[9, 1, 0] = 1
        # Third object:
        if self.num_objs > 2:
            labels_batch[0, 2, 2] = labels_batch[1, 2, 3] = 1
            labels_batch[2, 2, 4] = labels_batch[3, 2, 5] = 1
            labels_batch[4, 2, 6] = labels_batch[5, 2, 7] = 1
            labels_batch[6, 2, 8] = labels_batch[7, 2, 9] = 1
            labels_batch[8, 2, 0] = labels_batch[9, 2, 1] = 1
        # Fourth object:
        if self.num_objs > 3:
            labels_batch[0, 3, 3] = labels_batch[1, 3, 4] = 1
            labels_batch[2, 3, 5] = labels_batch[3, 3, 6] = 1
            labels_batch[4, 3, 7] = labels_batch[5, 3, 8] = 1
            labels_batch[6, 3, 9] = labels_batch[7, 3, 0] = 1
            labels_batch[8, 3, 1] = labels_batch[9, 3, 2] = 1

        bboxes_batch = self.bboxes_train[0:DUMP_SIZE]
        noises_batch = np.random.uniform(0, 1, size=(DUMP_SIZE, self.num_noises))
        render_batch = util.render_bboxes(bboxes_batch, labels_batch, shape=self.sample_shape)

        return [labels_batch, bboxes_batch, noises_batch, render_batch]

    def init_obj_dump_input(self):
        labels_batch = np.zeros((DUMP_SIZE, self.num_labels), dtype=np.float32)
        labels_batch[0, 0] = labels_batch[1, 1] = 1
        labels_batch[2, 2] = labels_batch[3, 3] = 1
        labels_batch[4, 4] = labels_batch[5, 5] = 1
        labels_batch[6, 6] = labels_batch[7, 7] = 1
        labels_batch[8, 8] = labels_batch[9, 9] = 1

        noises_batch = np.random.uniform(0, 1, size=(DUMP_SIZE, self.num_noises))

        return [labels_batch, noises_batch]


def main():
    trainer = Trainer()

    K.set_value(trainer.model.generator_optimizer.lr, 1e-4)
    K.set_value(trainer.model.discriminator_optimizer.lr, 1e-3)
    K.set_value(trainer.model.obj_generator_optimizer.lr, 1e-4)
    K.set_value(trainer.model.obj_discriminator_optimizer.lr, 1e-3)
    trainer.run_training(num_epochs=10000, start_at=0)
    util.save_model(trainer.model.gan, OUT_PATH, 'gan_0')
    util.save_model(trainer.model.obj_generator, OUT_PATH, 'obj_generator_0')
    util.save_model(trainer.model.obj_discriminator, OUT_PATH, 'obj_discriminator_0')

    K.set_value(trainer.model.generator_optimizer.lr, 1e-5)
    K.set_value(trainer.model.discriminator_optimizer.lr, 1e-4)
    K.set_value(trainer.model.obj_generator_optimizer.lr, 1e-5)
    K.set_value(trainer.model.obj_discriminator_optimizer.lr, 1e-4)
    trainer.run_training(num_epochs=5000, start_at=10000)
    util.save_model(trainer.model.gan, OUT_PATH, 'gan_1')
    util.save_model(trainer.model.obj_generator, OUT_PATH, 'obj_generator_1')
    util.save_model(trainer.model.obj_discriminator, OUT_PATH, 'obj_discriminator_1')

    K.set_value(trainer.model.generator_optimizer.lr, 1e-6)
    K.set_value(trainer.model.discriminator_optimizer.lr, 1e-5)
    K.set_value(trainer.model.obj_generator_optimizer.lr, 1e-6)
    K.set_value(trainer.model.obj_discriminator_optimizer.lr, 1e-5)
    trainer.run_training(num_epochs=5000, start_at=15000)
    util.save_model(trainer.model.gan, OUT_PATH, 'gan_2')
    util.save_model(trainer.model.obj_generator, OUT_PATH, 'obj_generator_2')
    util.save_model(trainer.model.obj_discriminator, OUT_PATH, 'obj_discriminator_2')

    util.plot_loss(LOSSES)


if __name__ == '__main__':
    main()
