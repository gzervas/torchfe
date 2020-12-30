import os
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="Number of observations")
parser.add_argument("-m", type=int, required=True, help="Number of datasets")
parser.add_argument("--datadir", type=str, required=True, help="Data directory")


def gen_data(n):
    rnd = np.random.default_rng()
    a, b = 1, 0.05

    fe_sizes = [n // 20, int(np.sqrt(n)), int(n ** 0.33)]
    fe = [rnd.choice(s + 1, n) for s in fe_sizes]
    fe_vals = [rnd.random(s + 1) for s in fe_sizes]

    x1 = rnd.random(n)
    x2 = x1 ** 2
    mu = a * x1 + b * x2

    for j in range(len(fe_vals)):
        mu += fe_vals[j][fe[j]]

    mu = np.exp(mu)
    y = rnd.negative_binomial(mu, 0.5)

    return np.log(y + 1), x1, fe


def main():
    args = parser.parse_args()

    for i in range(args.m):
        y, x, fe = gen_data(args.n)
        data = pd.DataFrame(
            dict(
                zip(
                    ["y", "x"] + [f"fe{i}" for i in range(len(fe))],
                    [y, x] + fe,
                )
            )
        )

        data.to_parquet(
            os.path.join(args.datadir, f"data_{i}.pq"),
            index=False,
        )


if __name__ == "__main__":
    main()
