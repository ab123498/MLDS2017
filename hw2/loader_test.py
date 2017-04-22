from loader import Loader
import numpy as np

loader = Loader()

for feat, cap in loader.train_data(10):
    feat = np.array(feat)
    print(feat.shape)
    print(cap)
    break
    
for id, feat in loader.test_data(10):
    print(feat.shape)
    print(id)
    print(feat[0])
    break
    
embedding = loader.get_embedding()
print(embedding.shape)
print(embedding[3:5])