import unittest

import numpy as np
import scipy.stats as stats


def my_entropy(px):
    return -np.sum(px * np.log2(px), axis=0)


def my_cross_entropy(p_true, p_predict):
    return -np.sum(p_true * np.log2(p_predict), axis=0)


# Also known as residual entropy.
def my_kullback_leibler(px, py):
    return np.sum(px * np.log2(px / py), axis=0)


def my_kullback_leibler2(px, py):
    return my_cross_entropy(px, py) - my_entropy(px)


def my_encoding_entropy(px, eb):
    return np.sum(px * eb)


#
# Unit Tests.
#

pk0 = np.asarray([.5, .5])
pk0_1 = np.asarray([.75, .25])
pk1 = np.asarray([.3, .4, .1, .2])
pk2 = np.asarray([.3, .4, .1, .2])
pk3 = np.asarray([.31, .39, .09, .21])

# Weather : Sun, Overcast, Sun+Cloud, Sun+HeavyCloud, LightRain, Rain, HeavyRain, Storm
w1 = np.asarray([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
w2 = np.asarray([.35, .35, .1, .1, 0.04, 0.04, 0.01, 0.01])
w3 = np.asarray([.01, .01, .04, .04, 0.1, 0.1, 0.35, 0.35])

# but used to encode each weather state
eb1 = np.asarray([3, 3, 3, 3, 3, 3, 3, 3])
eb2 = np.asarray([2, 2, 3, 3, 4, 4, 5, 5])


class TestEntropy(unittest.TestCase):

    def test_0(self):
        # two equally probable events = 1 bit of information
        self.assertAlmostEqual(my_entropy(pk0), 1)

        # Information is reduced if probabilities not equal, as in this case information is reduced
        # when we learn it is sunny as it sunny 75% of the time.
        self.assertAlmostEqual(my_entropy(pk0_1), 0.81, places=2)

        return

    def test_1(self):
        self.assertAlmostEqual(my_entropy(pk1), stats.entropy(pk=pk1, base=2))
        return

    # Divergence from self = 0
    def test_2(self):
        self.assertAlmostEqual(my_kullback_leibler(pk1, pk1), 0)
        return

    def test_3(self):
        self.assertAlmostEqual(my_kullback_leibler(pk1, pk2), stats.entropy(pk=pk1, qk=pk2, base=2))
        return

    def test_4(self):
        self.assertAlmostEqual(my_kullback_leibler(pk1, pk3), stats.entropy(pk=pk1, qk=pk3, base=2))
        return

    # Eight equally probable weather events = 3 bits
    def test_w0(self):
        self.assertAlmostEqual(my_entropy(w1), 3)
        return

    # Cross entropy is a function of bits used on average to encode a state. If less bits can be used
    # for most common states then cross entropy is closer to the entropy of the probability distribution
    # our ideal state is for cross entropy to be = entropy (lower limit)
    def test_w1(self):
        self.assertAlmostEqual(my_encoding_entropy(w2, eb2), 2.42, places=2)
        self.assertGreater(my_encoding_entropy(w2, eb2), my_entropy(w2))
        self.assertAlmostEqual(my_encoding_entropy(w3, eb2), 4.58, places=2)
        self.assertGreater(my_encoding_entropy(w3, eb2), my_entropy(w2))
        self.assertGreater(my_encoding_entropy(w3, eb2), my_encoding_entropy(w2, eb2))
        return

    # KL Divergence = Cross Entropy - Entropy.
    # So we test the difference between our expected distribution of weather events based on the
    # number of bits we chose to encode with vs. the actual distribution
    #
    def test_w2(self):
        p_true = w3
        p_predict = 1 / np.power(2, eb2)
        self.assertAlmostEqual(my_kullback_leibler2(p_true, p_predict), 2.35, places=2)
        return

    # Cross entropy loss function for NN classification.
    def test_cross_entropy_loss(self):
        p_one_hot = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
        p_nn_pred = np.asarray([0.1, 0.2, 0.4, 0.1, 0.2])
        self.assertAlmostEqual(my_cross_entropy(p_one_hot, p_nn_pred), -np.log2(.4))
        return


# Execute the UnitTests.
#


if __name__ == "__main__":
    tests = TestEntropy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
