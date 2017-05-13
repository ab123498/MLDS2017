import pandas as pd
import re
import numpy as np
import math
import scipy.misc
import skipthoughts

def train_data(batch_size):
    face_feat = ['eye', 'ear', 'nose',
                 'mouth', 'hair', 'tear',
                 'hat', 'male', 'female']

    face_feat_re = '|'.join([r'\b%ss?\b' % i for i in face_feat])


    df = pd.read_csv('MLDS_HW3_dataset/tags_clean.csv', header=None)
    df.columns = ['id', 'tags']

    def filt_tag(x):
        tags = [i.split(':')[0] for i in x.strip().split('\t')]

        tags = [i for i in tags if re.search(face_feat_re, i) or 'tail' in i]
        if len(tags) > 0:
            return ' '.join(tags)

    df.tags = df.tags.apply(filt_tag)
    null_df = df[df.tags.isnull()]
    null_df.reset_index(inplace=True, drop=True)

    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    n_batch = int(math.floor(len(df)/batch_size))

    encoder = skipthoughts.Encoder(skipthoughts.load_model())

    for i in range(n_batch):
        chunk = df[i*batch_size:(i+1)*batch_size]
        chunk_len = len(chunk)

        images = np.zeros([chunk_len, 64, 64, 3])
        for i, id in enumerate(chunk.id):
            image = scipy.misc.imread('./MLDS_HW3_dataset/faces/%d.jpg' % id)
            image = scipy.misc.imresize(image, (64, 64))
            images[i] = image

        wrong_images = np.zeros_like(images)
        for i, id in enumerate(null_df.sample(batch_size).id):
            image = scipy.misc.imread('./MLDS_HW3_dataset/faces/%d.jpg' % id)
            image = scipy.misc.imresize(image, (64, 64))
            wrong_images[i] = image

        captions = encoder.encode(chunk.tags.tolist(), verbose=False)
        
        yield images, wrong_images, captions


if __name__ == '__main__':
    next(train_data())
    