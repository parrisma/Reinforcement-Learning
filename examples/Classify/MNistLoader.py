import os
import struct
import unittest

import numpy as np


#
# based on https://gist.github.com/akesling/5358964
#   Which contains comment.
#   > Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
#   > which is GPL licensed.
#


class MNistLoader:

    @classmethod
    def read_mnist(cls,
                   training=True,
                   path="."):

        if training:
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        else:
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            _, _ = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        return img, lbl


#
# Unit Tests.
#
class TestMNISTLoader(unittest.TestCase):

    #
    # Test Image Load.
    #
    def test_0(self):
        ml = MNistLoader()
        img, lbl = ml.read_mnist(training=True,
                                 path="C:\\Users\\Admin_2\\Google Drive\\DataSets")
        s = np.shape(img)
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0], 60000)
        self.assertEqual(s[1], 28)
        self.assertEqual(s[2], 28)
        s = np.shape(lbl)
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0], 60000)
        return


#
# Execute the UnitTests.
#

if __name__ == "__main__":
    tests = TestMNISTLoader()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
