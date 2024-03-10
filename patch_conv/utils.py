from torch import nn

from .module import PatchConv2d


def convert_model(model: nn.Module, splits: int = 4) -> nn.Module:
    """
    Convert the convolutions in the model to PatchConv2d.
    """
    if isinstance(model, PatchConv2d):
        return model
    elif isinstance(model, nn.Conv2d) and model.kernel_size[0] > 1 and model.kernel_size[1] > 1:
        return PatchConv2d(splits=splits, conv2d=model)
    else:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, PatchConv2d)):
                continue
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d) and submodule.kernel_size[0] > 1 and submodule.kernel_size[1] > 1:
                    setattr(module, subname, PatchConv2d(splits=splits, conv2d=submodule))
        return model
