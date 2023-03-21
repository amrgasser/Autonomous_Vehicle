from IPython import display as ipythondisplay
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def display_model(model):
    tf.keras.utils.plot_model(model,
                              to_file='tmp.png',
                              show_shapes=True)
    return ipythondisplay.Image('tmp.png')


def plot_sample(x, y, vae):
    plt.figure(figsize=(2, 1))
    plt.subplot(1, 2, 1)

    idx = np.where(y == 1)[0][0]
    plt.imshow(x[idx])
    plt.grid(False)

    plt.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(recon[idx])
    plt.grid(False)

    plt.show()


def preprocess(full_obs, camera):
    # Extract ROI
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]

    # Rescale to [0, 1]
    obs = obs / 255.
    return obs


def grab_and_preprocess_obs(car, camera):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs, camera)
    return obs


def create_grid_of_images(xs, size=(5, 5)):
    grid = []
    counter = 0
    for i in range(size[0]):
        row = []
        for j in range(size[1]):
            row.append(xs[counter])
            counter += 1
        row = np.hstack(row)
        grid.append(row)
    grid = np.vstack(grid)
    return grid
