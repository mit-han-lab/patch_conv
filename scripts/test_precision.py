import torch
from torch import nn
from tqdm import tqdm

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
    batch_size: int,
    h: int,
    w: int,
    in_channels: int,
    output_channels: int,
    kernel_size: int,
    padding: int,
    stride: int,
    splits: int,
):
    model = Model(in_channels, output_channels, kernel_size, padding, stride, dtype=torch.float16, device="cuda")
    input = torch.randn(batch_size, in_channels, h, w, dtype=torch.float16, device="cuda")
    std_output = model(input)
    converted_model = convert_model(model, splits=splits)
    converted_output = converted_model(input)
    return torch.allclose(std_output, converted_output, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    in_channels = 32
    out_channels = 64

    cases = []

    for batch_size in [1, 2, 4]:
        for kernel_size, padding, stride in [(3, 1, 1), (3, 1, 2), (5, 2, 1), (5, 2, 2), (7, 3, 1), (7, 3, 2)]:
            for splits in [4, 8]:
                for h in range(1024, 4096, 256):
                    for w in range(1024, 4096, 256):
                        kwargs = {
                            "batch_size": batch_size,
                            "h": h,
                            "w": w,
                            "in_channels": in_channels,
                            "output_channels": out_channels,
                            "kernel_size": kernel_size,
                            "padding": padding,
                            "stride": stride,
                            "splits": splits,
                        }
                        cases.append(kwargs)
    for kwargs in tqdm(cases):
        check = test(**kwargs)
