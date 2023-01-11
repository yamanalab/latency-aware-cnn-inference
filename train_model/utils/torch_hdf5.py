import h5py
import torch.nn as nn

import constants

LINEAR_LAYER_CLASSES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)


def serialize_linear(module, f, base_key):
    key = '/'.join([base_key, 'weight'])
    f.create_dataset(key, data=module.weight.detach())

    if (module.bias is not None):
        key = '/'.join([base_key, 'bias'])
        f.create_dataset(key, data=module.bias.detach())


def serialize_batch_norm(module, f, base_key):
    key = '/'.join([base_key, 'weight'])
    f.create_dataset(key, data=module.weight.detach())

    key = '/'.join([base_key, 'bias'])
    f.create_dataset(key, data=module.bias.detach())

    key = '/'.join([base_key, 'running_mean'])
    f.create_dataset(key, data=module.running_mean)

    key = '/'.join([base_key, 'running_var'])
    f.create_dataset(key, data=module.running_var)


def serialize_group_norm(module, f, base_key):
    key = '/'.join([base_key, 'weight'])
    f.create_dataset(key, data=module.weight.detach())

    key = '/'.join([base_key, 'bias'])
    f.create_dataset(key, data=module.bias.detach())


def serialize(module, f, base_key, module_count_map):
    if isinstance(module, LINEAR_LAYER_CLASSES):
        if isinstance(module, nn.Conv2d):
            class_name = constants.CONV_2D_CLASS_NAME
            module_name = f'{class_name}_{module_count_map[class_name]}'
            serialize_linear(module, f, '/'.join([base_key, module_name]))
            module_count_map[class_name] += 1
        elif isinstance(module, nn.Linear):
            class_name = constants.LINEAR_CLASS_NAME
            module_name = f'{class_name}_{module_count_map[class_name]}'
            serialize_linear(module, f, '/'.join([base_key, module_name]))
            module_count_map[class_name] += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            class_name = constants.BATCH_NORM_CLASS_NAME
            module_name = f'{class_name}_{module_count_map[class_name]}'
            serialize_batch_norm(module, f, '/'.join([base_key, module_name]))
            module_count_map[class_name] += 1
        elif isinstance(module, nn.GroupNorm):
            class_name = constants.GROUP_NORM_CLASS_NAME
            module_name = f'{class_name}_{module_count_map[class_name]}'
            serialize_group_norm(module, f, '/'.join([base_key, module_name]))
            module_count_map[class_name] += 1
    else:
        for child in module.children():
            serialize(child, f, base_key, module_count_map)


def save_params_as_hdf5(module, filepath):
    module_count_map = {
        constants.CONV_2D_CLASS_NAME: 1,
        constants.AVG_POOL_2D_CLASS_NAME: 1,
        constants.BATCH_NORM_CLASS_NAME: 1,
        constants.GROUP_NORM_CLASS_NAME: 1,
        constants.ACTIVATION_CLASS_NAME: 1,
        constants.LINEAR_CLASS_NAME: 1,
        constants.FLATTEN_CLASS_NAME: 1
    }
    with h5py.File(filepath, 'w') as f:
        serialize(module, f, '', module_count_map)
