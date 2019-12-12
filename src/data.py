import os

import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


DATASET_PATH = '../data/2x2/train.npy'
SAMPLES_PATH = '../data/samples'
DATASET_SIZE = 60000


def save_image(canvas, labels, idx):
    img = canvas
    img *= 255
    img = Image.fromarray(img)
    img = img.convert('L')
    img_name = '_'.join(str(label) for label in labels)
    img.save(f'{SAMPLES_PATH}/{str(idx)}_{img_name}.png')


def draw_background(canvas):
    canvas = canvas.astype(np.uint8)
    canvas = iaa.Clouds().augment_image(canvas)
    canvas = canvas.astype(np.float32)
    canvas += 255 / 2
    canvas = np.clip(canvas, a_min=None, a_max=255)
    canvas /= 255

    return canvas


def draw_borders(canvas, grid_size, cell_size):
    grid_rows, grid_cols = grid_size
    cell_rows, cell_cols = cell_size

    # Draw row borders in black.
    for row in range(grid_rows - 1):
        idx = (row + 1) * cell_rows
        canvas[idx - 1, :] = 0
        canvas[idx, :] = 0

    # Draw col borders in black.
    for col in range(grid_cols - 1):
        idx = (col + 1) * cell_cols
        canvas[:, idx - 1] = 0
        canvas[:, idx] = 0

    return canvas


def make_dataset(grid_size, padding, draw_bg=False):
    grid_rows, grid_cols = grid_size
    pad_rows, pad_cols = padding

    # x_train: Grayscale black on white images of digits.
    #   dtype=uint8
    #   shape=(60000, 28, 28)
    #   values=[0, 255]
    # y_train: Labels of the digits.
    #   dtype=uint8
    #   shape=(60000,)
    #   values=[0, 9]
    (x_train, y_train), _ = mnist.load_data()

    # Cast to float, normalize values to [0, 1] and invert colors.
    x_train = x_train.astype(np.float32)
    x_train /= 255
    x_train = 1 - x_train

    samples = []
    num_saved = 0
    num_to_save = 100
    num_classes = 10
    num_cells = grid_rows * grid_cols

    num_objs = x_train.shape[0]
    obj_rows = x_train.shape[1]
    obj_cols = x_train.shape[2]

    cell_rows = pad_rows + obj_rows + pad_rows
    cell_cols = pad_cols + obj_cols + pad_cols

    canvas_rows = grid_rows * cell_rows
    canvas_cols = grid_cols * cell_cols

    for _ in range(0, DATASET_SIZE):
        take_idx = np.random.randint(0, num_objs, num_cells)
        objects = x_train[take_idx]
        labels = y_train[take_idx]
        bboxes = []

        # Create canvas.
        canvas = np.ones((canvas_rows, canvas_cols), dtype=np.float32)
        if draw_bg:
            canvas = draw_background(canvas)
        canvas = draw_borders(canvas, (grid_rows, grid_cols), (cell_rows, cell_cols))

        # Draw objects and save their bounding boxes.
        for row in range(grid_rows):
            for col in range(grid_cols):
                top = row * cell_rows + pad_rows
                left = col * cell_cols + pad_cols
                bottom = top + obj_rows
                right = left + obj_cols

                idx = row * grid_cols + col
                canvas[top:bottom, left:right] = np.minimum(canvas[top:bottom, left:right], objects[idx])
                bboxes.append([top, left, bottom, right])

        if num_saved < num_to_save:
            save_image(canvas, labels, num_saved)
            num_saved += 1

        # Add 3rd dimension because conv layers expect it.
        canvas = np.expand_dims(canvas, axis=3)
        # One-hot encode the labels.
        labels = to_categorical(labels, num_classes)

        samples.append([canvas, labels, bboxes])

    samples = np.array(samples)
    np.save(DATASET_PATH, samples)


def load_dataset():
    samples = np.load(DATASET_PATH, allow_pickle=True)

    # images: Grayscale black on white images of bordered tables with digits.
    #   dtype=float32
    #   shape=(DATASET_SIZE, 128, 128, 1)
    #   values=[0, 1]
    # labels: One-hot encoded label of each cell for all images.
    #   dtype=float32
    #   shape=(DATASET_SIZE, num_objs, 10)
    #   values=[0, 1]
    # bboxes: Bounding box (Top, Left, Bottom, Right) of each cell for all images.
    #   dtype=int32
    #   shape=(DATASET_SIZE, num_objs, 4)
    #   values=[0, 127]
    images = np.array([x[0] for x in samples], dtype=np.float32)
    labels = np.array([x[1] for x in samples], dtype=np.float32)
    bboxes = np.array([x[2] for x in samples], dtype=np.int32)

    return images, labels, bboxes


if __name__ == '__main__':
    if not os.path.exists(SAMPLES_PATH):
        os.makedirs(SAMPLES_PATH)

    make_dataset((2, 2), (18, 18), True)
