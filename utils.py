import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def apply_mask(op, x, valid_lengths):
    if op._get_name() == 'Conv1d':
        kernel_size = op.kernel_size[0]
        stride = op.stride[0]
        padding = op.padding[0]
        dilation = op.dilation[0] 
        output_length = []
        for i, length in enumerate(valid_lengths):
            output_length.append((length + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1)
            x[i, :, output_length[i]:] = 0
    elif op._get_name() == 'ConvTranspose1d':
        kernel_size = op.kernel_size[0]
        stride = op.stride[0]
        padding = op.padding[0]
        dilation = op.dilation[0]
        output_length = []
        for i, length in enumerate(valid_lengths):
            output_length.append((length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + op.output_padding[0] + 1)
            x[i, :, output_length[i]:] = 0
    else:
        return x, valid_lengths
    return x, output_length

