import os.path
import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as PLT


def load_images():
    images = list()
    id = 0
    n = 0
    xs = ys = 50
    fn = "src_img/" + str(n) + ".bmp"
    while n < 1000:
        if os.path.exists(fn):
            im = Image.open(fn)
            im = im.resize((xs, ys), Image.LANCZOS)
            ima = np.array(im.getdata(), np.int32).reshape(im.size[1], im.size[0], 3)
            images.append(ima)
            id += 1
        fn = "src_img/" + str(n) + ".bmp"
        n += 1
    imgs = np.zeros((id, xs, ys, 3), dtype=np.int32)
    id = 0
    for i in images:
        imgs[id] = i
        id += 1
    return imgs


if __name__ == '__main__':
    ima = load_images()
    for i in ima:
        PLT.imshow(i)
        PLT.show()
        time.sleep(.5)
    exit(0)
