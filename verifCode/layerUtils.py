import torch
import torch.nn as nn
import numpy as np
from im2colUtils import *

def weightCompar(myWeights, myBiases, fromPyTorch):
    return (myWeights - fromPyTorch.weight.grad.data).abs().max(), (myBiases - fromPyTorch.bias.grad.data).abs().max()
 
def dataReader(batchSize):
    transform = transforms.Compose([transforms.ToTensor(), ])
    dataGetter = lambda isTrain: torchvision.datasets.MNIST(root='./MNIST-data/', train=isTrain, transform=transform, download=True)
    trainingSet, testingSet = dataGetter(isTrain=True), dataGetter(isTrain=False)
    testingImages, testingLabels = iter(torch.utils.data.DataLoader(testingSet, batch_size=testingSet.__len__(), shuffle=False)).next()
    return torch.utils.data.DataLoader(trainingSet, batch_size=batchSize, shuffle=True), Variable(testingImages), testingLabels

def makeIm2Col(forwardTensor, layer):
    if isinstance(forwardTensor, torch.FloatTensor):
        forwardTensor = forwardTensor.data.numpy()
    im2col = im2col_indices(forwardTensor, *getLayerProperties(layer))
    return {'im2col': im2col, 'maxIndices': np.argmax(im2col, axis=0)}

def convolution(forwardTensor, layer, weightsParam):
    batchSize, depthIn, heightIn, widthIn = forwardTensor.shape
    sizeOut = subSample(heightIn, *getLayerProperties(layer))
    numberOfFilters, filterDepth, filterHeight, filterWidth = weightsParam.shape
    assert filterHeight == filterWidth and filterDepth == depthIn
    im2col = makeIm2Col(forwardTensor, layer).get('im2col')
    weightsCol = weightsParam.reshape(numberOfFilters, -1)
    outData = np.dot(weightsCol, im2col)
    outData = outData.reshape(numberOfFilters, sizeOut, sizeOut, batchSize)
    return outData.transpose(3, 0, 1, 2), im2col

def normBN2d(tensor):
    assert tensor.dim() == 2
    return (tensor - tensor.mean(dim=0)) / tensor.std(dim=0, unbiased=False)

def bn2d(tensor):
    batchSize, numbFeatures, resolution, _ = tensor.size()
    return tensor.permute(0, 2, 3, 1).reshape(batchSize * resolution * resolution, numbFeatures)

def bn4d(tensor, batchSize):
    if type(tensor) is np.ndarray:
        tensor = torch.from_numpy(tensor).float()
    effectiveBatchSize, numbFeatures = tensor.size()
    resolution = int(np.sqrt(effectiveBatchSize / batchSize))
    return tensor.reshape(batchSize, resolution, resolution, numbFeatures).permute(0, 3, 1, 2).contiguous()
