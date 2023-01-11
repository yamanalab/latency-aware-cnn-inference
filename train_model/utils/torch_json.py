import json
import re

import constants


def construct_structure_list(module, structure_list, module_count_map):
    module_name = module._get_name()
    if re.search('^Conv2d$', module_name):
        class_name = constants.CONV_2D_CLASS_NAME
        stride = [module.stride, module.stride] if type(module.stride) is int else list(module.stride)
        padding = module.padding if type(module.padding) is str else [module.padding, module.padding] if type(module.padding) is int else list(module.padding)
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'filters': module.weight.size()[0],
                'kernel_size': [module.weight.size()[2], module.weight.size()[3]],
                'stride': stride,
                'padding': padding
            }
        })
        module_count_map[class_name] += 1
    elif re.search('^AvgPool2d$', module_name):
        class_name = constants.AVG_POOL_2D_CLASS_NAME
        kernel_size = [module.kernel_size, module.kernel_size] if type(module.kernel_size) is int else list(module.kernel_size)
        stride = [module.stride, module.stride] if type(module.stride) is int else list(module.stride)
        padding = [module.padding, module.padding] if type(module.padding) is int else list(module.padding)
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding
            }
        })
        module_count_map[class_name] += 1
    elif re.search('^Linear$', module_name):
        class_name = constants.FLATTEN_CLASS_NAME
        if module_count_map[class_name] == 1:
            structure_list.append({
                'class_name': class_name,
                'info': {
                    'name': f'{class_name}_{module_count_map[class_name]}',
                }
            })
            module_count_map[class_name] += 1
        class_name = constants.LINEAR_CLASS_NAME
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'units': module.weight.size()[0]
            }
        })
        module_count_map[class_name] += 1
    elif re.search('^BatchNorm.*$', module_name):
        class_name = constants.BATCH_NORM_CLASS_NAME
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'eps': str(module.eps)
            }
        })
        module_count_map[class_name] += 1
    elif re.search('^GroupNorm.*$', module_name):
        class_name = constants.GROUP_NORM_CLASS_NAME
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'eps': str(module.eps)
            }
        })
        module_count_map[class_name] += 1
    elif re.search('^ReLU.*$|^Relu.*$|^Swish.*$|^Mish.*$|^Square$|^GELU.*$', module_name):
        class_name = constants.ACTIVATION_CLASS_NAME
        structure_list.append({
            'class_name': class_name,
            'info': {
                'name': f'{class_name}_{module_count_map[class_name]}',
                'function': module_name
            }
        })
        module_count_map[class_name] += 1
    else:
        for child in module.children():
            construct_structure_list(child, structure_list, module_count_map)


def save_structure_as_json(module, filepath):
    structure_list = []
    structure_dict = {}
    module_count_map = {
        constants.CONV_2D_CLASS_NAME: 1,
        constants.AVG_POOL_2D_CLASS_NAME: 1,
        constants.BATCH_NORM_CLASS_NAME: 1,
        constants.GROUP_NORM_CLASS_NAME: 1,
        constants.ACTIVATION_CLASS_NAME: 1,
        constants.LINEAR_CLASS_NAME: 1,
        constants.FLATTEN_CLASS_NAME: 1
    }
    construct_structure_list(module, structure_list, module_count_map)
    structure_dict['class_name'] = module._get_name()
    structure_dict['structure'] = structure_list
    with open(filepath, 'w') as f:
        json.dump(structure_dict, f, indent=2)
