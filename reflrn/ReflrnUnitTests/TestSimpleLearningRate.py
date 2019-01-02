import unittest
import random

from reflrn.SimpleLearningRate import SimpleLearningRate


class TestSimpleLearningRate(unittest.TestCase):
    tolerance_4dp = 4
    tolerance_9dp = 9

    def test_simple(self):
        slr = SimpleLearningRate(lr0=1,
                                 lrd=0.9,
                                 lr_min=0)
        test_cases = [[1, 0.5263],
                      [5, 0.1818],
                      [40, 0.0270],
                      [100, 0.011]
                      ]
        for case in test_cases:
            step, expected = case
            lr = slr.learning_rate(step)
            self.assertAlmostEqual(lr, expected, self.tolerance_4dp)
        return

    def test_min(self):
        lr_min = 0.1
        slr = SimpleLearningRate(lr0=1,
                                 lrd=0.9,
                                 lr_min=lr_min)
        test_cases = [[1, max(lr_min, 0.5263)],
                      [5, max(lr_min, 0.1818)],
                      [40, max(lr_min, 0.0270)],
                      [100, max(lr_min, 0.011)]
                      ]
        for case in test_cases:
            step, expected = case
        lr = slr.learning_rate(step)
        self.assertAlmostEqual(lr, expected, self.tolerance_4dp)
        return

    def test_target(self):
        lr0 = 1
        lr_min = 0
        for s in range(1, 50):
            lrt = random.uniform(lr_min, lr0)
            lrd = SimpleLearningRate.lr_decay_target(learning_rate_zero=lr0,
                                                     target_step=s,
                                                     target_learning_rate=lrt)
            slr = SimpleLearningRate(lr0=lr0,
                                     lrd=lrd,
                                     lr_min=lr_min)

            lr = slr.learning_rate(step=s)
            self.assertAlmostEqual(lr, lrt, self.tolerance_9dp)


if __name__ == "__main__":
    tests = TestSimpleLearningRate()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
