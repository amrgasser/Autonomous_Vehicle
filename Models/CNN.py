import tensorflow as tf
import functools
from Models.IModel import IModel
import tensorflow_probability as tfp
# Defines a CNN with 4 layers of convolution to define steering angle
# and outputs a normal probability distribution


class CNN(IModel):
    def __init__(self) -> None:
        super().__init__()
        self.driving_model = self.create_driving_model()

    def create_driving_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2),
            tf.keras.layers.Conv2D(filters=38, kernel_size=5, strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=128, activation=tf.keras.activations.swish),
            tf.keras.layers.Dense(units=2, activation=None)
        ])
        return model

    def run_driving_model(self, image):
        # Arguments:
        #   image: an input image
        # Returns:
        #   pred_dist: predicted distribution of control actions
        single_image_input = tf.rank(image) == 3  # missing 4th batch dimension
        if single_image_input:
            image = tf.expand_dims(image, axis=0)

        distribution = self.driving_model(image)

        mu, logsigma = tf.split(distribution, 2, axis=1)
        mu = self.max_curvature * tf.tanh(mu)  # conversion
        sigma = self.max_std * tf.sigmoid(logsigma) + 0.005  # conversion

        pred_dist = tfp.distributions.Normal(mu, sigma)

        return pred_dist
