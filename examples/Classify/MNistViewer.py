import unittest
from random import randint
from typing import List

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot

from examples.Classify.MNistLoader import MNistLoader


class MNistViewer:
    @classmethod
    def view_img(cls, image=List[float]) -> None:
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()
        return


#
# Unit Tests.
#


class TestMNISTViewer(unittest.TestCase):

    #
    # Test Image Load.
    #
    def test_0(self):
        ml = MNistLoader()
        img, lbl = ml.read_mnist(training=True,
                                 path="C:\\Users\\Admin_2\\Google Drive\\DataSets")
        s = np.shape(img)
        simg = img[randint(0,s[0])]
        MNistViewer.view_img(simg)
        return

    #
    # Execute the UnitTests.
    #


if __name__ == "__main__":
    tests = TestMNISTViewer()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
