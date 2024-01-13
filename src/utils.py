import numpy as np


def generate_rand_emb(x, y):
    rng = np.random.default_rng(seed=42)

    # generate initial set
    e = {tuple(rng.binomial(1, 0.5, x)) for i in range(y)}

    while len(e) < y:
        e.add(rng.binomial(1, 0.5, x))

    return np.array(tuple(e))
