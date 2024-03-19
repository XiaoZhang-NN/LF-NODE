import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dropblock import DropBlock

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim, kernel_size):
        super(ODEfunc, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = ConcatConv2d(dim, dim, kernel_size, 1, (kernel_size//2))
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = ConcatConv2d(dim, dim, kernel_size, 1, (kernel_size//2))
        # self.conv3 = ConcatConv2d(dim, dim, kernel_size, 1, (kernel_size//2))
        # self.norm3 = nn.BatchNorm2d(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(t, x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm2(out)
        # # out = self.relu(out)

        # out = self.conv3(t, out)
        # out = self.norm3(out)
        # import pdb;pdb.set_trace()
        return out

##############################kernel_size=3,5 结合TEM,异步耦合    2    ###################
class ODEBlock_GAC(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock_GAC, self).__init__()
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
class ODEBlock_ODE(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock_ODE, self).__init__()
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
               # out = dhdt *self.args.step_size * self.args.func_size
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
        self.relu = nn.LeakyReLU(0.1)
        self.norm2 = nn.BatchNorm2d(self.input_feats)
        self.norm1 = nn.BatchNorm2d(n_feats)

    def forward(self, t, x):
        k1 = self.conv1(t, x) #16,64,48,48
        k1 = self.norm1(k1)
        k1 = self.relu(k1) #16,69,48,48
        k2 = self.conv2(t, k1) #16,69,48,48
        k = self.norm2(k2)
        # k = self.relu(k2) #16,64,48,48
        return k

class ODEBlock_AUGODE(nn.Module):

    def __init__(self, opt, dim):
        super(ODEBlock_AUGODE, self).__init__()
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

class Meta_ODE_TEM(nn.Module):
    def __init__(self, channels, opt, dim):
        super( Meta_ODE_TEM, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ODEBlock_ODE(opt, dim))
        self.dncnn_ode = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.dncnn_ode(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        return out

class Meta_AUGODE_TEM(nn.Module):
    def __init__(self, channels, opt, dim):
        super( Meta_AUGODE_TEM, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ODEBlock_AUGODE(opt, dim))
        self.dncnn_augode = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.dncnn_augode(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        return out

class Meta_GAC(nn.Module):
    def __init__(self, channels, opt, dim):
        super(Meta_GAC, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ODEBlock_GAC(opt, dim))
        self.dncnn_gac = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.dncnn_gac(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        return out

