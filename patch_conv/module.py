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
            split_size = h // self.splits
            x_permuted = x.view(b, c, self.splits, split_size, w).permute(0, 2, 1, 3, 4)
            padding_bak = self.conv2d.padding
            self.conv2d.padding = (0, self.conv2d.padding[1])
            output = torch.zeros(b, self.splits, self.conv2d.out_channels, split_size + 2 * self.conv2d.padding[0], w, device=x.device)
            for i in range(self.splits):
                if i == 0:
                    x_padded = F.pad(
                        x_permuted[:, i],
                        (0, 0, self.conv2d.padding[0], self.conv2d.padding[0]),
                        mode="constant" if self.conv2d.padding_mode == "zeros" else self.conv2d.padding_mode,
                        value=0,
                    )
                else:
                    x_padded[:, :, : self.conv2d.padding[0]] = output[:, i - 1, :, -2 * self.conv2d.padding[0] : -self.conv2d.padding[0]]
                    x_padded[:, :, -self.conv2d.padding[0] :] = x_permuted[:, i, :, : self.conv2d.padding[0]]
                output[:, i] = self.conv2d(x_padded, *args, **kwargs)
            self.conv2d.padding = padding_bak
            output = output.permute(0, 2, 1, 3, 4).reshape(b, self.conv2d.out_channels, -1, w)
            return output
        else:
            return self.conv2d(x, *args, **kwargs)
