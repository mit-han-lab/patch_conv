import torch
from torch import nn
from torch.nn import functional as F


class PatchConv2d(nn.Module):
    def __init__(self, splits: int = 4, conv2d: nn.Conv2d = None, *args, **kwargs):
        super(PatchConv2d, self).__init__()
        if conv2d is not None:
            self.conv2d = conv2d
            self.splits = splits
        else:
            self.conv2d = nn.Conv2d(*args, **kwargs)
            self.splits = splits

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, c, h, w = x.shape
        if c * h * w >= (1 << 30):
            assert h % self.splits == 0
            x_permuted = x.view(b, c, self.splits, h // self.splits, w).permute(0, 2, 1, 3, 4)
            x_padded = F.pad(
                x_permuted,
                (0, 0, self.conv2d.padding[0], self.conv2d.padding[0]),
                mode="constant" if self.conv2d.padding_mode == "zeros" else self.conv2d.padding_mode,
                value=0,
            )
            x_padded[:, 1:, :, : self.conv2d.padding[0]] = x_permuted[:, :-1, :, -self.conv2d.padding[0] :]
            x_padded[:, :-1, :, -self.conv2d.padding[0] :] = x_permuted[:, 1:, :, : self.conv2d.padding[0]]
            x_padded = x_padded.view(b * self.splits, c, -1, w)
            padding_bak = self.conv2d.padding
            self.conv2d.padding = (0, self.conv2d.padding[1])
            output = self.conv2d(x_padded, *args, **kwargs)
            self.conv2d.padding = padding_bak
            _, oc, oh, ow = output.shape
            output = output.view(b, self.splits, oc, oh, ow).permute(0, 2, 1, 3, 4).reshape(b, oc, -1, ow)
            return output
        else:
            return self.conv2d(x, *args, **kwargs)
