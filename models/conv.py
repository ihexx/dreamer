import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd

from .base import Module

class ConvEncoder(Module):

    def __init__(self, depth=32, act=tf.nn.relu):
        self._act = act
        self._depth = depth

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
        x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
        x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
        shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape)


class ConvDecoder(Module):

    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        self._act = act
        self._depth = depth
        self._shape = shape

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
