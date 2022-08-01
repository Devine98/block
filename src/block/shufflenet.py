# -*- coding: utf-8 -*-
"""
Created on 20200824
@author: sz

torch shufflenet v2


"""



##deep 
import torch
import torch.nn as nn,torch.nn.functional as F
import torch.optim as optim
cuda = torch.cuda.is_available()    
device = 'cuda' if cuda else 'cpu'
n_device = torch.cuda.device_count()


######################################shuffle net


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x




class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit,self).__init__()
        mid_channels = out_channels // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, 
                          in_channels, 
                          kernel_size = 3, 
                          stride=stride, 
                          padding=1, 
                          groups=in_channels, 
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, 
                          mid_channels, 
                          kernel_size = 1, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, 
                          mid_channels, 
                          kernel_size = 1, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 
                          mid_channels, 
                          kernel_size = 3, 
                          stride=stride, 
                          padding=1, 
                          groups=mid_channels, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels,
                          mid_channels, 
                          kernel_size = 1, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, 
                          mid_channels, 
                          kernel_size = 1, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 
                          mid_channels, 
                          kernel_size = 3, 
                          stride=stride, 
                          padding=1, 
                          groups=mid_channels, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, 
                          mid_channels, 
                          kernel_size = 1, 
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        self.stride = stride
    def forward(self, x):
        if self.stride == 1:
            x1, x2 =  x.chunk(2, dim=1)
            out = torch.cat((self.branch1(x1), 
                             self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), 
                             self.branch2(x)), dim=1)
        out = shuffle_channels(out, 2)
        return out
    
    
    
class ShuffleNetV2(nn.Module):
    def __init__(self, multiplier = 2):
        super().__init__()
        map_dict = {
            0.5 : [48,96,192,1024],
            1 : [116,232,464,1024],
            1.5:[176,352,704,1024],
            2 : [244,488,976,2048],
        }
        channel_num = map_dict[multiplier]
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels = 4, 
                          out_channels = 24, 
                          kernel_size = 3, 
                          stride=2, 
                          padding=1, 
                          bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                )
 
        self.maxpool = nn.MaxPool2d(kernel_size=3, 
                                    stride=2, 
                                    padding=1)
 
        self.stage2 = self.make_layers(24, channel_num[0], 
                                       layers_num=4, stride=2)
        self.stage3 = self.make_layers(channel_num[0], 
                                       channel_num[1], 
                                       layers_num=8, stride=2)
        self.stage4 = self.make_layers(channel_num[1], 
                                       channel_num[2], 
                                       layers_num=4, stride=2)
 
        self.conv5 = nn.Sequential(
                nn.Conv2d(channel_num[2], 
                          channel_num[3], 1, bias=False),
                nn.BatchNorm2d(channel_num[3]),
                nn.ReLU(inplace=True)
        )
 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel_num[3], 32)
        #self.fc2 = nn.Linear(32, 32)
 
    def make_layers(self, in_channels, 
                    out_channels, layers_num, 
                    stride=2):
        layers = []
        layers.append(ShuffleUnit(in_channels, 
                                  out_channels, 
                                  stride=stride))
 
        for i in range(layers_num - 1):
            layers.append(ShuffleUnit(out_channels, 
                                      out_channels, 
                                      stride=1))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        # x : b,4,256,256 
        x = self.conv1(x)
        # x : b , 24 , 128 ,128 
        x = self.maxpool(x)
        # x : b , 24 , 64 ,64 
        x = self.stage2(x)
        # x : b , 176 , 32 ,32 
        x = self.stage3(x)
        # x : b , 352 ,16 ,16 
        x = self.stage4(x)
        # x : b , 704 , 8 ,8 
        x = self.conv5(x)
        # x : b , 1024 , 8 , 8 
        x = self.avgpool(x)
        # x : b , 1024 ,1 , 1 
        x = x.flatten(1)
        # x : b ,1024 
        x = self.fc1(x)
        # x : b ,32 
        return x
    
    

    def init_weight(self):
        for m in self.modules():
            #全连接层参数初始化
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                #m.weight.data.normal_(mean=0,std=2)        
            if isinstance(m,nn.Conv2d):
    #             nn.init.normal(m.weight.data)
#                 nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data)#卷积层参数初始化
    
        
 
