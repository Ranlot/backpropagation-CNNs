import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from layerUtils import *
from im2colUtils import *
from backPropUtils import *
# --------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(stride=2, padding=0, kernel_size=2)

        self.bn1 = nn.BatchNorm2d(6, eps=0., track_running_stats=False, affine=True)
        self.bn2 = nn.BatchNorm2d(16, eps=0., track_running_stats=False, affine=True)
        self.bn3 = nn.BatchNorm1d(120, eps=0., track_running_stats=False, affine=True)
        self.bn4 = nn.BatchNorm1d(84, eps=0., track_running_stats=False, affine=True)

        self.fc1 = nn.Linear(in_features=(16 * 4 * 4), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.activation = nn.Tanh() if True else nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, A0):
        A1 = self.conv1(A0)
        A2 = self.activation(A1)
        A3 = self.bn1(A2)
        A4 = self.pool(A3)
        A5 = self.conv2(A4)
        A6 = self.activation(A5)
        A7 = self.bn2(A6)
        A8 = self.pool(A7)
        A9 = A8.view(-1, 16 * 4 * 4)
        A10 = self.fc1(A9)
        A11 = self.activation(A10)
        A12 = self.bn3(A11)
        A13 = self.fc2(A12)
        A14 = self.activation(A13)
        A15 = self.bn4(A14)
        A16 = self.fc3(A15)
        P = self.softmax(A16)

        A0im2col = makeIm2Col(A0, self.conv1)
        A3im2col = makeIm2Col(A3.view(A3.size()[0] * 6, 1, 24, 24).detach().numpy(), self.pool)
        A4im2col = makeIm2Col(A4.detach().numpy(), self.conv2)
        A6im2col = makeIm2Col(A6.view(A6.size()[0] * 16, 1, 8, 8).detach().numpy(), self.pool)

        return {'A0': {'data': A0.data, 'im2col': A0im2col}, 'A1': A1.data, 'A2': A2.data, 'A3': {'data': A3.data, 'maxIndices': A3im2col['maxIndices']}, 'A4': {'data': A4.data, 'im2col': A4im2col}, 'A5': A5.data, 'A6': {'data': A6.data, 'maxIndices': A6im2col['maxIndices']}, 'A7': A7.data, 'A8': A8.data, 'A9': A9.data, 'A10': A10.data, 'A11': A11.data, 'A12': A12.data, 'A13': A13.data, 'A14': A14.data, 'A15': A15, 'A16': A16, 'P': P.data}
# --------------------------------------------------------

batchSize = 60
trainLoader, testingImages, testingLabels = dataReader(batchSize)

convNet = ConvNet()
convNet.double()

optimizer = optim.SGD(convNet.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(40):
    for miniBatchID, miniBatch in enumerate(trainLoader):
        images, labels = [Variable(_) for _ in miniBatch]
        images = images.double()

        optimizer.zero_grad()
        convNet.zero_grad()
        tensors = convNet.forward(images)

        loss = criterion(tensors['A16'], labels)
        loss.backward() 

        if np.random.uniform() > 0.98: # check manual derivatives before weight update
            delta16 = deltaLoss(tensors['P'], labels, batchSize)
            delta15 = deltaFC(delta16, convNet.fc3)
            delta14 = deltaBN(delta15, tensors['A14'], convNet.bn4, batchSize)
            delta13 = deltaActivation(delta14, tensors['A13'], convNet.activation)
            delta12 = deltaFC(delta13, convNet.fc2)
            delta11 = deltaBN(delta12, tensors['A11'], convNet.bn3, batchSize)
            delta10 = deltaActivation(delta11, tensors['A10'], convNet.activation)
            delta9 = deltaFC(delta10, convNet.fc1)
            delta8 = delta9.view(batchSize, 16, 4, 4).detach().numpy()
            delta7 = deltaMaxPool(delta8, tensors['A6'], convNet.pool)

            delta7 = torch.from_numpy(delta7).double()
            delta7 = bn2d(delta7)
            A6_2d = bn2d(tensors['A6']['data'])
            delta6 = deltaBN(delta7, A6_2d, convNet.bn2, batchSize * 8 * 8)
            delta6 = bn4d(delta6, batchSize).detach().numpy()

            delta5 = deltaActivation(torch.from_numpy(delta6).double(), tensors['A5'], convNet.activation).numpy()
            delta4 = deltaConv(delta5, convNet.conv2)

            delta3 = deltaMaxPool(delta4, tensors['A3'], convNet.pool)

            delta3 = torch.from_numpy(delta3).double()
            delta3 = bn2d(delta3)
            A2_2d = bn2d(tensors['A2'])
            delta2 = deltaBN(delta3, A2_2d, convNet.bn1, batchSize * 24 * 24)
            delta2 = bn4d(delta2, batchSize)

            delta1 = deltaActivation(delta2.double(), tensors['A1'], convNet.activation).numpy()

            myDw15, myDb15 = updateFC(delta16, tensors['A15'])
            myDw14, myDb14 = updateBN(delta15, tensors['A14'], batchSize)
            myDw12, myDb12 = updateFC(delta13, tensors['A12'])
            myDw11, myDb11 = updateBN(delta12, tensors['A11'], batchSize)
            myDw9, myDb9 = updateFC(delta10, tensors['A9'])
            myDw6, myDb6 = updateBN(delta7, A6_2d, batchSize) 
            myDw4, myDb4 = updateConv(delta5, tensors['A4'])
            myDw2, myDb2 = updateBN(delta3, A2_2d, batchSize)
            myDw0, myDb0 = updateConv(delta1, tensors['A0'])

            errDw15, errDb15 = weightCompar(myDw15, myDb15, convNet.fc3)
            errDw14, errDb14 = weightCompar(myDw14, myDb14, convNet.bn4)           
            errDw12, errDb12 = weightCompar(myDw12, myDb12, convNet.fc2)
            errDw11, errDb11 = weightCompar(myDw11, myDb11, convNet.bn3)        
            errDw9, errDb9 = weightCompar(myDw9, myDb9, convNet.fc1)
            errDw6, errDb6 = weightCompar(myDw6, myDb6, convNet.bn2)
            errDw4, errDb4 = weightCompar(myDw4, myDb4, convNet.conv2)
            errDw2, errDb2 = weightCompar(myDw2, myDb2, convNet.bn1)
            errDw0, errDb0 = weightCompar(myDw0, myDb0, convNet.conv1)

            print('\nepoch = %d ; minibatch id = %d' % (epoch, miniBatchID))
            print('errDw15 = %.4g' % errDw15); print('errDb15 = %.4g' % errDb15)
            print('errDw14 = %.4g' % errDw14); print('errDb14 = %.4g' % errDb14)
            print('errDw12 = %.4g' % errDw12); print('errDb12 = %.4g' % errDb12)
            print('errDw11 = %.4g' % errDw11); print('errDb11 = %.4g' % errDb11)
            print('errDw9 = %.4g' % errDw9); print('errDb9 = %.4g' % errDb9)
            print('errDw6 = %.4g' % errDw6); print('errDb6 = %.4g' % errDb6)
            print('errDw4 = %.4g' % errDw4); print('errDb4 = %.4g' % errDb4)
            print('errDw2 = %.4g' % errDw2); print('errDb2 = %.4g' % errDb2)
            print('errDw0 = %.4g' % errDw0); print('errDb0 = %.4g' % errDb0)
            print('\n')

        optimizer.step() # do the actual weight update according to selected optimizer
