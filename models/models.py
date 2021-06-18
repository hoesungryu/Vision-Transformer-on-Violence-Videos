# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T


#------------------------------# 
# ResNet(2+1)D
#------------------------------# 
class Transfer_r2plus1d_18(nn.Module):
    def __init__(self):
        super(Transfer_r2plus1d_18, self).__init__()
        self.model = self.freeze()

    def freeze(self):
        model = models.video.r2plus1d_18(pretrained=True)
        
        unfreeze_layer = ['layer4.1.conv2.1.weight','layer4.1.conv2.1.bias',
                          'fc.weight',' fc.bias']
        
        for name, param in model.named_parameters():
            if name in unfreeze_layer:
                param.requires_grad = True
            else: 
                param.requires_grad = False
                
        model.fc = nn.Linear(512, 1)
        return model 
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
#------------------------------# 
# VVT
#------------------------------#     
class TransfomerEncoder_MLP(nn.Module):
    def __init__(self):
        super(TransfomerEncoder_MLP, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=16*16, nhead=16,)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4,) 
        
        self.avp = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536,32)
        self.fc2 = nn.Linear(32,1)
        self.dropout= nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.transpose(0,1)
        x = self.transformer_encoder(x)
        x = x.transpose(0,1)
        x = self.avp(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        return x

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        return x


class Separable_Extractor(nn.Module):
    def __init__(self):
        super(Separable_Extractor, self).__init__()
        self.pretrain_extractor = self.freeze()
        
    def freeze(self):
        model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        num_classes = 16*16
        in_features = model._fc.in_features
        # model._fc = nn.Linear(in_features,num_classes)
        model._fc = nn.Sequential(nn.Linear(in_features, 512),
                                  nn.Dropout(0.2),
                                 nn.Linear(512,num_classes))
        return model
    
    def forward(self, x):
        embeded_vectors= []
#         print(x.shape) #[4, 24, 3, 224, 224]
        for frame in range(x.size(1)):
#             tmp = self.pretrain_extractor(x[:,:,frame,:,:].squeeze(2))
#             print(frame)
#             print(x[:,:,frame,:,:].shape)
            tmp = self.pretrain_extractor(x[:,frame,:,:,:])
            embeded_vectors.append(tmp)

        embeded_vectors = torch.stack(embeded_vectors,dim=1)
        
        return embeded_vectors # shape=(Batch, frame, width, height) = (4, 24,256)    
    
