"""__init__.py."""

import pickle

import cloudpickle

# Use cloudpickle's Pickler in order to enable pickling of instnace methods
# in multiprocessing
pickle.Pickler = cloudpickle.Pickler
