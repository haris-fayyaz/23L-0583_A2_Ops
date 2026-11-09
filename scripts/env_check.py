import sys
import platform
from datetime import datetime

import matplotlib
import numpy
import pandas
import PIL
import sklearn
import tensorflow as tf

print("Time        :", datetime.now().isoformat(timespec="seconds"))
print("OS          :", platform.platform())
print("Python      :", sys.version.split()[0])
print("TensorFlow  :", tf.__version__)
print("Keras       :", tf.keras.__version__)
print("NumPy       :", numpy.__version__)
print("Pandas      :", pandas.__version__)
print("Pillow      :", PIL.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPUs        :", tf.config.list_physical_devices("GPU"))
