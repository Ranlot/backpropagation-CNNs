from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch

def imageViewer(imageTensor, label):
    plt.figure()
    plt.gray()
    plt.imshow(imageTensor.numpy()[0])
    plt.title('class = %s' % label)
    plt.savefig('tt.png')

def getLayerProperties(layer):
        if type(layer) == torch.nn.modules.conv.Conv2d:
                return layer.kernel_size[0], layer.padding[0], layer.stride[0]
        else:
                assert type(layer) == torch.nn.modules.pooling.MaxPool2d
                return layer.kernel_size, layer.padding, layer.stride

def subSample(sizeIn, kernel, padding, stride):
        assert (sizeIn + 2 * padding - kernel) % stride == 0
        return int((sizeIn + 2 * padding - kernel) / stride + 1)

def get_im2col_indices(x_shape, kernel, padding, stride):
    N, C, H, W = x_shape
    sizeOut = subSample(H, kernel, padding, stride)
    i0 = np.repeat(np.arange(kernel), kernel)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(sizeOut), sizeOut)
    j0 = np.tile(np.arange(kernel), kernel * C)
    j1 = stride * np.tile(np.arange(sizeOut), sizeOut)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kernel * kernel).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, kernel, padding, stride):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, kernel, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel * kernel * C, -1)
    return cols
