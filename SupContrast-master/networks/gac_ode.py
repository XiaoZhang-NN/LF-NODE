"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.BatchNorm2d(dim) #nn.GroupNorm(min(int(dim/2.0), dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim, kernel_size):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, kernel_size, 1, (kernel_size//2))
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, kernel_size, 1, (kernel_size//2))
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
##############################kernel_size=3,5 结合TEM,异步耦合    2    ###################
class ODEBlock_coupling(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock_coupling, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim, 3).cuda(),ODEfunc(dim, 5).cuda()])
        self.alphafc = nn.Sequential(nn.Linear(opt.num_block, opt.hidden_dim), nn.Tanh(), nn.Linear(opt.hidden_dim, opt.num_block))
        self.args = opt

    def forward(self, x):
        num_step = int(self.args.depth/self.args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(self.args.num_block).cuda() / self.args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % self.args.coupling_func == 0:
                for j in range(self.args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * self.args.step_size * self.args.func_size
            if i % self.args.coupling_alpha == 0:
                alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * self.args.step_size  * self.args.alpha_size
            t = t + self.args.step_size
        return out
##############################TEM###################
class ODEBlock1(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock1, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim, 3).cuda() for _ in range(opt.num_block)])
        self.alphafc = nn.Sequential(nn.Linear(opt.num_block, opt.hidden_dim), nn.Tanh(), nn.Linear(opt.hidden_dim, opt.num_block))
        self.args = opt
    def forward(self, x):
        num_step = int(self.args.depth/self.args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(self.args.num_block).cuda() / self.args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % self.args.coupling_func == 0:
                for j in range(self.args.num_block):
                    # import pdb;pdb.set_trace()
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt *self.args.step_size * self.args.func_size
            if i % self.args.coupling_alpha == 0:
                alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * self.args.step_size * self.args.alpha_size
            t = t + self.args.step_size
        return out
        ##############################AUG###################
class augodefunc(nn.Module):
    def __init__(
        self, n_feats, augment_feats, kernel_size ):

        super(augodefunc, self).__init__()
        self.input_feats = n_feats + augment_feats
        self.augment_feats = augment_feats
        self.conv1 =  ConcatConv2d(self.input_feats, n_feats, kernel_size, 1, (kernel_size//2))
        self.conv2 =  ConcatConv2d(n_feats, self.input_feats, kernel_size, 1, (kernel_size//2))
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm(self.input_feats)
        self.norm2 = norm(n_feats)
        self.norm3 = norm(self.input_feats)

    def forward(self, t, x):
        k1 = self.norm1(x)
        k1 = self.relu(k1) #16,69,48,48
        k2 = self.conv1(t, k1) #16,64,48,48
        k2 = self.norm2(k2)
        k3 = self.relu(k2) #16,64,48,48
        k4 = self.conv2(t, k3) #16,69,48,48
        k = self.norm3(k4)
        # import pdb;pdb.set_trace()
        return k

class ODEBlock_aug_func(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock_aug_func, self).__init__()
        self.odefunc = nn.ModuleList([augodefunc(dim,augment_feats=4, kernel_size=3).cuda(), augodefunc(dim,augment_feats=4, kernel_size=3).cuda()])
        self.alphafc = nn.Sequential(nn.Linear(opt.num_block, opt.hidden_dim), nn.Tanh(), nn.Linear(opt.hidden_dim, opt.num_block))
        self.args = opt

    def forward(self, x):
        num_step = int(self.args.depth/self.args.step_size)
        batch_size, channels, height, width = x.shape
        aug = torch.zeros(batch_size, 4,
                                height, width).cuda()
        x_aug = torch.cat([x, aug], 1).cuda()
        out = x_aug; t = 0.0
        alpha = torch.ones(self.args.num_block).cuda() / self.args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % self.args.coupling_func == 0:
                for j in range(self.args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * self.args.step_size * self.args.func_size
            alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * self.args.step_size * self.args.alpha_size
            t = t + self.args.step_size
        return out

class odenet(nn.Module):
    def __init__(self, opt, block, in_channel=3):
        super(odenet, self).__init__()
        self.in_planes = 64

        #self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.conv1 =nn.Conv2d(3, 64, 3, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu =nn.ReLU(inplace=True)
        self.conv2 =nn.Conv2d(64, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 =nn.Conv2d(64, 64, 4, 2, 1)
        self.block = block(opt, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.conv3(self.relu(self.bn2(self.conv2(out))))
        out = self.block(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

def resnet18(opt):
    return odenet( opt, ODEBlock1)# ODEBlock1(opt,64) #ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(opt):
    return odenet( opt, ODEBlock_coupling)#ODEBlock_coupling(opt,64)  #ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(opt):
    return odenet( opt,ODEBlock_aug_func)#ODEBlock_aug_func(opt,64) #ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

#def resnet101(**kwargs):
#    return ODEBlock_aug_func(64) #ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 64],
    'resnet34': [resnet34, 64],
    'resnet50': [resnet50, 64],
   # 'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, opt, name='resnet18', head='mlp', feat_dim=64):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        if name=='resnet50':
            dim_in = dim_in + 4
            feat_dim = feat_dim + 4
        opt = opt
        self.encoder = model_fun(opt)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        # import pdb;pdb.set_trace()
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet18', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        if name=='resnet50':
            # dim_in = dim_in + 4
            feat_dim = feat_dim + 4
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
