import torch
import torch.nn as nn

ACTIVATIONS = ['relu', 'swish', 'mish', 'square',
               'relu_rg4_deg2', 'relu_rg5_deg2', 'relu_rg6_deg2', 'relu_rg7_deg2',      # ReLU deg2 approx.
               'relu_rg4_deg4', 'relu_rg5_deg4', 'relu_rg6_deg4', 'relu_rg7_deg4',      # ReLU deg4 approx.
               'swish_rg4_deg2', 'swish_rg5_deg2', 'swish_rg6_deg2', 'swish_rg7_deg2',  # Swish deg2 approx.
               'swish_rg4_deg4', 'swish_rg5_deg4', 'swish_rg6_deg4', 'swish_rg7_deg4',  # Swish deg4 approx.
               'mish_rg4_deg2', 'mish_rg5_deg2', 'mish_rg6_deg2', 'mish_rg7_deg2',      # Mish deg2 approx
               'mish_rg4_deg4', 'mish_rg5_deg4', 'mish_rg6_deg4', 'mish_rg7_deg4']      # Mish deg4 approx.


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


#########################################################
# ReLU approximation
#########################################################
# 2 degree
class ReluRg4Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.375373 + 0.5 * x + 0.117071 * x**2


class ReluRg5Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.47 + 0.5 * x + 0.09 * x**2


class ReluRg6Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.563059 + 0.5 * x + 0.078047 * x**2


class ReluRg7Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.6569 + 0.5 * x + 0.0669 * x**2


# 4 degree
class ReluRg4Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.234606 + 0.5 * x + 0.204875 * x**2 - 0.0063896 * x**4


class ReluRg5Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.2932 + 0.5 * x + 0.1639 * x**2 - 0.00327 * x**4


class ReluRg6Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.119782 + 0.5 * x + 0.147298 * x**2 - 0.0020115 * x**4


class ReluRg7Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.41056 + 0.5 * x + 0.11707 * x**2 - 0.00119 * x**4


#########################################################
# Swish approximation
#########################################################
# 2 degree
class SwishRg4Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.145276 + 0.5 * x + 0.12592 * x**2


class SwishRg5Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.24 + 0.5 * x + 0.1 * x**2


class SwishRg6Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.344125 + 0.5 * x + 0.085105 * x**2


class SwishRg7Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.4517 + 0.5 * x + 0.0723 * x**2


# 4 degree
class SwishRg4Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.03347 + 0.5 * x + 0.19566 * x**2 - 0.005075 * x**4


class SwishRg5Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.07066 + 0.5 * x + 0.17 * x**2 - 0.00315 * x**4


class SwishRg6Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.1198 + 0.5 * x + 0.1473 * x**2 - 0.002012 * x**4


class SwishRg7Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.1773 + 0.5 * x + 0.128 * x**2 - 0.001328 * x**4


#########################################################
# Mish approximation
#########################################################
# 2 degree
class MishRg4Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.194373 + 0.525933 * x + 0.1268256 * x**2


class MishRg5Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.2967 + 0.516 * x + 0.1 * x**2


class MishRg6Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.402808 + 0.510062 * x + 0.0837098 * x**2


class MishRg7Deg2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5095 + 0.5066 * x + 0.071 * x**2


# 4 degree
class MishRg4Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.06021 + 0.565775 * x + 0.21051 * x**2 - 0.004142 * x**3 - 0.00609 * x**4


class MishRg5Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.1104 + 0.5495 * x + 0.17573 * x**2 - 0.00346 * x**4


class MishRg6Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.169 + 0.53663 * x + 0.148529 * x**2 - 0.001277 * x**3 - 0.002096 * x**4


class MishRg7Deg4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.232356 + 0.527 * x + 0.12748 * x**2 - 0.00134 * x**4
