import argparse
import json
import os
import time

import torch
from torch import nn
from tqdm import tqdm, trange

from patch_conv import convert_model


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        dtype: torch.dtype,
        device: str,
    ):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        return self.conv(x)


def test(
    h: int,
    w: int,
    in_channels: int,
    output_channels: int,
    kernel_size: int,
    padding: int,
    stride: int,
    splits: int,
    warmup_times=5,
    test_times=20,
    use_patch_conv: bool = False,
    ignore_ratio: float = 0.2,
):
    model = Model(in_channels, output_channels, kernel_size, padding, stride, dtype=torch.float16, device="cuda")
    input = torch.randn(1, in_channels, h, w, dtype=torch.float16, device="cuda")
    if use_patch_conv:
        model = convert_model(model, splits=splits)
    for _ in trange(warmup_times, position=1, leave=False, desc="Warmup"):
        model(input)
        torch.cuda.synchronize()
    latencies = []
    for _ in trange(test_times, position=1, leave=False, desc="Test"):
        start_time = time.time()
        model(input)
        torch.cuda.synchronize()
        latencies.append(time.time() - start_time)
    sorted(latencies)
    ignored_num = int(len(latencies) * ignore_ratio / 2)
    if ignored_num > 0:
        latencies = latencies[ignored_num:-ignored_num]
    latency = sum(latencies) / len(latencies)
    return latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_patch_conv", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    cases = []

    for kernel_size, padding, stride in [(3, 1, 1), (5, 2, 1), (7, 3, 1)]:
        for splits in [4]:
            for s, c in [(4096, 64)]:
                kwargs = {
                    "h": s,
                    "w": s,
                    "in_channels": c,
                    "output_channels": c,
                    "kernel_size": kernel_size,
                    "padding": padding,
                    "stride": stride,
                    "splits": splits,
                    "use_patch_conv": args.use_patch_conv,
                }
                cases.append(kwargs)
    results = []
    for kwargs in tqdm(cases, position=0, leave=False, desc="Case"):
        latency = test(**kwargs)
        results.append((kwargs, latency * 1000))
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
