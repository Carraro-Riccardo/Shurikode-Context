import math
from torchvision import models
from torch import nn
from shurikodecontext.utils import ModelType, DeviceType


import torch
from importlib.resources import files

model_selector = {
    "r18": ("ResNet18", models.resnet18),
    "r34": ("ResNet34", models.resnet34),
    "r50": ("ResNet50", models.resnet50),
}


def Create_ResNet_Shurikode(
    model_type: ModelType = "r50",
    out_features: int = 256,
    device: DeviceType = "cpu",
    binary_output=False,
    hamming_output=False,
    group_norm=False,
):
    """
    Given the model type, the output_features and a device on which load the model, returns the ResNet model configured
    in the desired way.

    :param model_type: The type of ResNet. Available choices are 'r18', 'r34' and 'r50'.
    :param out_features: The number of output features. Corresponds to the possible number of labels.
    :param device: The type of device on which to load the model. Available choices are 'cpu', 'cuda' and 'mps'.
    :param binary_output: Wether if the model outputs a binary encoded number.
    :param hamming_output: Wether if to use hamming correction.
    """
    assert (
        binary_output or not hamming_output
    ), "You can't apply hamming correction on a model with a non-binary output."

    # New output feature calculation
    output_features = out_features
    if binary_output:
        output_features = math.ceil(math.log(out_features, 2))
        if hamming_output:
            correction_bits = 0
            # Determine the number of parity bits needed
            while (2**correction_bits) < (output_features + correction_bits + 1):
                correction_bits += 1
            output_features += correction_bits

    str_model, func_model = model_selector[model_type]
    model = func_model()
    model.fc = nn.Linear(model.fc.in_features, output_features)
    checkpoint_path = files("shurikodecontext.ml_model").joinpath(f"{str_model}.pth.tar")
    state_dict = torch.load(
        str(checkpoint_path), map_location=torch.device(device), weights_only=True
    )
    if group_norm:
        replace_batchnorm_with_groupnorm(model, 8)
    model.load_state_dict(state_dict)
    return model.to(device)


def replace_batchnorm_with_groupnorm(model: nn.Module, num_groups=8):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.GroupNorm(num_groups, module.num_features))
        else:
            replace_batchnorm_with_groupnorm(module, num_groups)
