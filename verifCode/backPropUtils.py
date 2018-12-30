import torch
import torch.nn as nn
import numpy as np
from layerUtils import convolution

def deltaLoss(predictedProbabilities, groundTruthLabels, batchSize):
    groundTruthLabels = torch.DoubleTensor(batchSize, 10).zero_().scatter_(1, groundTruthLabels.view(-1, 1).data, 1)
    return predictedProbabilities - groundTruthLabels

def deltaFC(upstreamGrad, fcLayer):
    return upstreamGrad @ (fcLayer.weight.data)

def deltaBN(upstreamGrad, inputData, bnLayer, effectiveBatchSize):
    delta = (upstreamGrad * bnLayer.weight).detach().numpy()
    dataSTD = inputData.std(dim=0, unbiased=False)
    normalizedInput = ((inputData - inputData.mean(dim=0)) / dataSTD).detach().numpy()
    return (effectiveBatchSize * delta - np.sum(delta, axis=0) - normalizedInput * np.sum(normalizedInput * delta, axis=0)) / (effectiveBatchSize * dataSTD).detach().numpy()
    
def deltaActivation(upstreamGrad, tensor, activationLayer):
    activationType = type(activationLayer)
    if activationType == torch.nn.modules.activation.Tanh:
        derivNonLinear = lambda x : 1. / (np.cosh(x) ** 2)
        return upstreamGrad * derivNonLinear(tensor)
    else:
        assert activationType == torch.nn.modules.activation.ReLU
        return upstreamGrad * torch.from_numpy(np.where(tensor.numpy() > 0, 1, 0)).double()

def deltaConv(upstreamGrad, convLayer):
    rotatedFilters = convLayer.weight.data.numpy().transpose(1, 0, 2, 3)[:,:,::-1,::-1]
    relevantConv = nn.Conv2d(in_channels=convLayer.out_channels, out_channels=convLayer.in_channels, kernel_size=convLayer.kernel_size[0], stride=1, padding=(convLayer.kernel_size[0]-1))
    return convolution(upstreamGrad, relevantConv, rotatedFilters)[0]

def deltaMaxPool(upstreamGrad, forwardTensor, maxPoolLayer):
    maxIndicesForward, downStreamGradSize = forwardTensor['maxIndices'], forwardTensor['data'].shape[-1]
    batchSize, numbFeatureMaps, featureMapsSize = upstreamGrad.shape[:3]
    kernelSize = maxPoolLayer.kernel_size
    unfold_batch_featureMaps = batchSize * numbFeatureMaps
    upstreamGrad_featureMaps = featureMapsSize * featureMapsSize
    downStreamGradSize = (batchSize, numbFeatureMaps, downStreamGradSize, downStreamGradSize)
    upstreamGrad = upstreamGrad.reshape(unfold_batch_featureMaps, upstreamGrad_featureMaps)
    maxIndicesForward = maxIndicesForward.reshape(upstreamGrad_featureMaps, unfold_batch_featureMaps).T
    def upSampleSingleCell(_):
        a = np.zeros(kernelSize * kernelSize)
        a[_[0]] = _[1]
        return a
    def singleFeatureMap(featureMap):
        _ = map(upSampleSingleCell, zip(maxIndicesForward[featureMap], upstreamGrad[featureMap]))
        return np.vstack(map(np.hstack, np.reshape(list(_), (featureMapsSize, featureMapsSize, kernelSize, kernelSize))))
    return np.reshape([singleFeatureMap(_) for _ in range(unfold_batch_featureMaps)], downStreamGradSize)

def updateFC(deltaAtThisLayer, tensor):
    batchSize = deltaAtThisLayer.shape[0]
    gradientBiases = (torch.ones(batchSize).double().unsqueeze(0)) @ deltaAtThisLayer
    gradientWeights = tensor.t() @ deltaAtThisLayer
    return gradientWeights.t() / batchSize, gradientBiases / batchSize

def updateConv(deltaAtThisLayer, tensor):
    batchSize, depthIn, widthIn, heightIn = deltaAtThisLayer.shape
    depthOut = tensor.get('data').shape[1]
    reshapedDelta = deltaAtThisLayer.transpose(1, 2, 3, 0).reshape(depthIn, widthIn * heightIn * batchSize)
    gradientWeights = np.dot(reshapedDelta, tensor.get('im2col').get('im2col').T)
    inferredKernelSize = int(np.sqrt(gradientWeights.shape[1] / depthOut))
    gradientWeights = gradientWeights.reshape(depthIn, depthOut, inferredKernelSize, inferredKernelSize)
    gradientBiases = np.dot(reshapedDelta, np.ones((widthIn * heightIn * batchSize, 1)))[:, 0]
    return torch.from_numpy(gradientWeights / batchSize).double(), torch.from_numpy(gradientBiases / batchSize).double()

def updateBN(deltaAtThisLayer, tensor, batchSize):
    gradientBiases = torch.sum(deltaAtThisLayer, dim=0) / batchSize
    normData = (tensor - tensor.mean(dim=0)) / tensor.std(dim=0, unbiased=False)
    gradientWeights = torch.diag(normData.transpose(1, 0) @ deltaAtThisLayer) / batchSize
    return gradientWeights, gradientBiases
