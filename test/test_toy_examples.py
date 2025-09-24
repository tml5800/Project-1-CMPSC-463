from src.loader import make_synthetic_mixture
from src.similarity import corr_dist
from src.kadane import max_activity_window


def test_kadane_nonempty():
x = [0,1,2,3,2,1,0]
s,e,val = max_activity_window(x)
assert e >= s


def test_similarity_sanity():
s = make_synthetic_mixture(6, 128, n_groups=2, seed=3)
d_same = corr_dist(s[0], s[2])
d_diff = corr_dist(s[0], s[3])
assert d_same < d_diff