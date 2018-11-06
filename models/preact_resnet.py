'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBlockUP(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, pixelshuffle=True, dropout_p=0.0):
        super(PreActBlockUP, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Sequential(
            nn.PixelShuffle(stride),
            nn.Conv2d(in_planes // (stride ** 2), planes, kernel_size=3, stride=1, padding=1, bias=False)
        ) if pixelshuffle else (nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False) if stride > 1
                                else nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.PixelShuffle(stride),
                nn.Conv2d(in_planes//(stride**2), self.expansion*planes, kernel_size=1, stride=1, bias=False)
            ) if pixelshuffle else (nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False) if stride > 1
                                else nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.drop = nn.Dropout2d(dropout_p)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        out = self.drop(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActBottleneckUP(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, pixelshuffle=True):
        super(PreActBottleneckUP, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Sequential(
            nn.PixelShuffle(stride),
            nn.Conv2d(planes // (stride ** 2), planes, kernel_size=3, stride=1, padding=1, bias=False)
        ) if pixelshuffle else (nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) if stride > 1
                                else nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.PixelShuffle(stride),
                nn.Conv2d(in_planes // (stride ** 2), self.expansion * planes, kernel_size=1, stride=1, bias=False),
            ) if pixelshuffle else (nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) if stride > 1
                                else nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_paths=2):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.num_paths = num_paths
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        block_up = PreActBottleneckUP if isinstance(block, PreActBottleneck) else PreActBlockUP
        self.dropout_p = 0.3
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=2),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block_up, 256, num_blocks[2], stride=2),
            self._make_layer(block_up, 128, num_blocks[1], stride=2),
            self._make_layer(block_up, 64, num_blocks[0], stride=2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh())

        self.path2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=2),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block_up, 256, num_blocks[2], stride=2),
            self._make_layer(block_up, 128, num_blocks[1], stride=2),
            self._make_layer(block_up, 64, num_blocks[0], stride=2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh())

        self.in_planes = 64
        self.drop = nn.Dropout2d(self.dropout_p)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if isinstance(block, PreActBlockUP):
                layers.append(block(self.in_planes, planes, stride, dropout_p=self.dropout_p))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        before_paths = []

        all_logits = torch.zeros((self.num_paths, x.size(0), self.num_classes)).to(self.device)
        for pathi in range(self.num_paths):
            if pathi == 0:
                before_this_path = self.path1(x)
            else:
                before_this_path = self.path2(x)
            before_paths.append(before_this_path)
            layer0 = self.conv1(self.drop(before_this_path))
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)
            pool = F.avg_pool2d(layer4, 4)
            pool = pool.view(pool.size(0), -1)
            logits = self.linear(pool)
            all_logits[pathi] = logits

        return all_logits, before_paths

def PreActResNet10():
    return PreActResNet(PreActBlock, [1,1,1,1])

def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


# def test():
#     net = PreActResNet18()
#     y = net((torch.randn(1,3,32,32)))
# print(y.size())