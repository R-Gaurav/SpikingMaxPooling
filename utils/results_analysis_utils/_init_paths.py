import os.path as osp
import sys
import tensorflow as tf

# Set memory growth on GPU.
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

path = osp.join(this_dir, "../..")
add_path(path)
