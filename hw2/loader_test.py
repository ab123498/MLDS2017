from loader import Loader
import numpy as np

loader = Loader()

for feat, cap in loader.train_data(10):
    feat = np.array(feat)
    print(feat.shape)
    print(cap)
    break