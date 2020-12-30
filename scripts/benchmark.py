import os
import argparse
import pandas as pd
import torchfe
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, required=True, help="Data directory")
parser.add_argument("--outfile", type=str, required=True, help="Benchmark file")


def main():
    assert torch.cuda.is_available()

    args = parser.parse_args()
    bench = pd.DataFrame()

    # hack to init pytorch
    __ = torch.zeros(1, device="cuda")

    for root, __, files in os.walk(args.datadir, topdown=False):
        for name in files:
            fn = os.path.join(root, name)
            data = pd.read_parquet(fn)
            print(fn)
            fe = data.filter(regex=r"^fe", axis=1)

            beta, loss, t1, t2 = torchfe.fit(
                data.y.values, data.x.values, fe.values.T)

            bench = bench.append({
                "filename": fn,
                "n": data.shape[0],
                "nfe": fe.shape[1],
                "beta": beta.item(),
                "loss": loss.item(),
                "t_secs": t2,
                "t_secs_gpu_xfer": t1,
            }, ignore_index=True)

    bench.to_csv(args.outfile)


if __name__ == "__main__":
    main()
