import h5py as h5
import os
import tqdm
import numpy as np


def fraction(path, name, every):
    f = h5.File(os.path.join(path, name))
    for i in tqdm.tqdm(range(1, 6)):
        name = list(f.keys())[i]
        b = f[name].value[range(549364)[0::every]]
        del f[name]
        f.create_dataset(name, data=b)

    a = np.array(f.attrs['split'])
    for i in range(len(f.attrs['split']) - 1):
        a[i][3] = len(f['label'])

    f.attrs.modify('split', a)


if __name__ == "__main__":
    fraction('/var/data/users/local/kchledowski/data/snli', 'train10.h5', 10)
