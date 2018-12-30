import unittest

from reflrn.CorrelationId import CorrelationId


class TestCorrelationId(unittest.TestCase):

    def test_simple(self):
        cid = CorrelationId()
        self.assertTrue(isinstance(cid.id(), str))
        self.assertTrue(len(cid.id()) > 0)

    def test_correlation_id_same_across_objects(self):
        cid1 = CorrelationId()
        cid2 = CorrelationId()
        self.assertTrue(cid1.__eq__(cid2))


if __name__ == "__main__":
    tests = TestCorrelationId()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
