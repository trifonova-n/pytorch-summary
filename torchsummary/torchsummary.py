import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


class Summary(OrderedDict):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.batch_size = batch_size

    def __str__(self):
        str_list = []
        str_list.append("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        str_list.append(line_new)
        str_list.append("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in self:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(self[layer]["output_shape"]),
                "{0:,}".format(self[layer]["nb_params"]),
            )
            total_params += self[layer]["nb_params"]
            total_output += np.prod(self[layer]["output_shape"])
            if "trainable" in self[layer]:
                if self[layer]["trainable"] == True:
                    trainable_params += self[layer]["nb_params"]
            str_list.append(line_new)

        # assume 4 bytes/number (float on cuda).
        input_sizes = [np.prod(in_size) for in_size in self.input_size]
        total_input_size = abs(np.prod(input_sizes) * self.batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        str_list.append("================================================================")
        str_list.append("Total params: {0:,}".format(total_params))
        str_list.append("Trainable params: {0:,}".format(trainable_params))
        str_list.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
        str_list.append("----------------------------------------------------------------")
        str_list.append("Input size (MB): %0.2f" % total_input_size)
        str_list.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
        str_list.append("Params size (MB): %0.2f" % total_params_size)
        str_list.append("Estimated Total Size (MB): %0.2f" % total_size)
        str_list.append("----------------------------------------------------------------")
        return '\n'.join(str_list)

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = Summary()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()


    return summary
