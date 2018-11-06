from models.cnn import SimpleModel
from models.preact_resnet import PreActResNet10, PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.wresnet import WideResNet

class ModelSelector():
    """
    My own changes include:
        Probability of cutout
        max/min size, again, probabilistic
        width and height differences
        intensity diferences
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,
                 dataset='Cifar-10',
                 in_shape=(32,32,3),
                 filters=(64, 128, 256, 512),
                 activation='leaky_relu',
                 widen_factor=2,
                 num_classes=10,
                 resdepth=18,
                 ):
        self.dataset = dataset
        self.filters = filters
        self.activation = activation
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.resdepth = resdepth
        self.in_shape = in_shape

    def select(self, model, path_fc=False):
        if model == 'cnn':
            net = SimpleModel(in_shape=self.in_shape,
                              activation=self.activation,
                              num_classes=self.num_classes,
                              filters=self.filters,
                              )
        else:
            assert (self.dataset != 'MNIST' and self.dataset != 'Fashion-MNIST'), "Cannot use resnet or densenet for mnist style data"
            if model == 'resnet':
                assert self.resdepth in [18, 34, 50, 101, 152], "Non-standard and unsupported resnet depth ({})".format(self.resdepth)
                if self.resdepth == 18:
                    net = ResNet18()
                elif self.resdepth == 34:
                    net = ResNet34()
                elif self.resdepth == 50:
                    net = ResNet50()
                elif self.resdepth == 101:
                    net = ResNet101()
                else:
                    net = ResNet152()
            elif model == 'densenet':
                assert self.resdepth in [121, 161, 169, 201], "Non-standard and unsupported densenet depth ({})".format(self.resdepth)
                if self.resdepth == 121:
                    net = DenseNet121()
                elif self.resdepth == 161:
                    net = DenseNet161()
                elif self.resdepth == 169:
                    net = DenseNet169()
                else:
                    net = DenseNet201()
            elif model == 'preact_resnet':
                assert self.resdepth in [10, 18, 34, 50, 101,
                                         152], "Non-standard and unsupported preact resnet depth ({})".format(self.resdepth)
                if self.resdepth == 10:
                    net = PreActResNet10(path_fc=path_fc)
                elif self.resdepth == 18:
                    net = PreActResNet18()
                elif self.resdepth == 34:
                    net = PreActResNet34()
                elif self.resdepth == 50:
                    net = PreActResNet50()
                elif self.resdepth == 101:
                    net = PreActResNet101()
                else:
                    net = PreActResNet152()
            elif model == 'wresnet':
                assert ((self.resdepth - 4) % 6 == 0), "Wideresnet depth of {} not supported, must fulfill: (depth - 4) % 6 = 0".format(self.resdepth)
                net = WideResNet(depth=self.resdepth,
                                 num_classes=self.num_classes,
                                 widen_factor=self.widen_factor)

        return net