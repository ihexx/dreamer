import pathlib
import pickle
import tensorflow as tf

class Module(tf.Module):

    def save(self, filename):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        with pathlib.Path(filename).open('wb') as f:
            pickle.dump(values, f)

    def load(self, filename):
        with pathlib.Path(filename).open('rb') as f:
            values = pickle.load(f)
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]