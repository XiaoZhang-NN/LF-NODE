import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)

parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type = str, choices=['euler', 'midpoint','dopri5','adaptive_heun'], default = 'euler')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--func', type=str, choices=['odetem','coupling','coupling2','temfunc'], default='odetem')
parser.add_argument('--num_block', type = int, default = 3)
parser.add_argument('--step_size', type=float, default=0.05)
parser.add_argument('--depth', type=float, default=1.0)
parser.add_argument('--hidden_dim', type=int, default=1000)
parser.add_argument('--coupling', type=int, default=2)
parser.add_argument('--coupling_func', type=int, default=2)
parser.add_argument('--coupling_alpha', type=int, default=1)
parser.add_argument('--func_size', type=float, default=1.0)
parser.add_argument('--alpha_size', type=float, default=1.0)


args = parser.parse_args()

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


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
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

##############################kernel_size=1,3,5,7,9 结合TEM,异步耦合###################
class ODEBlock_coupling(nn.Module):

    def __init__(self, dim):
        super(ODEBlock_coupling, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim, 3).to(device),ODEfunc(dim, 5).to(device),ODEfunc(dim, 7).to(device)])
        self.alphafc = nn.Sequential(nn.Linear(args.num_block, args.hidden_dim), nn.Tanh(), nn.Linear(args.hidden_dim, args.num_block))

    def forward(self, x):
        num_step = int(args.depth/args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(args.num_block).to(device) / args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % args.coupling_func == 0:
                for j in range(args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * args.step_size * args.func_size
            if i % args.coupling_alpha == 0:
                alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * args.step_size  * args.alpha_size
            t = t + args.step_size
        return out
##############################kernel_size=1,3,5,7,9 结合TEM,异步耦合    2    ###################
class ODEBlock_coupling2(nn.Module):

    def __init__(self, dim):
        super(ODEBlock_coupling2, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim, 3).to(device),ODEfunc(dim, 5).to(device)])#,ODEfunc(dim, 7).to(device)
        self.alphafc = nn.Sequential(nn.Linear(args.num_block, args.hidden_dim), nn.Tanh(), nn.Linear(args.hidden_dim, args.num_block))

    def forward(self, x):
        num_step = int(args.depth/args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(args.num_block).to(device) / args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % args.coupling_func == 0:
                for j in range(args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * args.step_size * args.func_size
            if i % args.coupling_alpha == 0:
                alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * args.step_size  * args.alpha_size
            t = t + args.step_size
        return out
##############################kernel_size=1,3,5,7,9 结合TEM,异步耦合###################
class ODEBlock_tem_func(nn.Module):

    def __init__(self, dim):
        super(ODEBlock_tem_func, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim, 3).to(device), ODEfunc(dim, 5).to(device), ODEfunc(dim, 7).to(device)])
        self.alphafc = nn.Sequential(nn.Linear(args.num_block, args.hidden_dim), nn.Tanh(), nn.Linear(args.hidden_dim, args.num_block))

    def forward(self, x):
        num_step = int(args.depth/args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(args.num_block).to(device) / args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % args.coupling == 0:
                for j in range(args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * args.step_size * 2
            alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * args.step_size
            t = t + args.step_size
        return out

##############################TEM###################
class ODEBlock(nn.Module):

    def __init__(self, dim):
        super(ODEBlock, self).__init__()
        self.odefunc = nn.ModuleList([ODEfunc(dim,3).to(device) for _ in range(args.num_block)])
        self.alphafc = nn.Sequential(nn.Linear(args.num_block, args.hidden_dim), nn.Tanh(), nn.Linear(args.hidden_dim, args.num_block))

    def forward(self, x):
        num_step = int(args.depth/args.step_size)
        out = x; t = 0.0
        alpha = torch.ones(args.num_block).to(device) / args.num_block
        for i in range(num_step):
            dhdt = 0
            weight = F.softmax(alpha, dim=-1)
            if i % args.coupling_func == 0:
                for j in range(args.num_block):
                    dhdt = dhdt + self.odefunc[j](t,out) * weight[j]
                out = out + dhdt * args.step_size * args.func_size
            if i % args.coupling_alpha == 0:
                alpha = alpha + self.alphafc(alpha.unsqueeze(0)).squeeze() * args.step_size  * args.alpha_size
            t = t + args.step_size
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def get_cifar100_loaders(data_aug=False, batch_size=128, test_batch_size=128, perc=1.0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
            normalize,
    ])

    train_loader = DataLoader(
        datasets.CIFAR100(root='/data1/XIAO_XIAO/NODE/example-y0-noise/data', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=8, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR100(root='/data1/XIAO_XIAO/NODE/example-y0-noise/data', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True
    )

    return train_loader, test_loader, None


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    correct_1 = 0.0
    correct_5 = 0.0
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)

        _, pred = output.topk(5, 1, largest=True, sorted=True)
        label = y.view(y.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        ####compute top1
        correct_1 += correct[:,:1].sum()
        ####compute top5
        correct_5 += correct[:,:5].sum()

    return correct_1 / len(dataset_loader.dataset), correct_5 / len(dataset_loader.dataset)

def accuracy_all(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 100)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):  # 判断是否需要创建文件夹,存在则跳过
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger
from thop import profile

if __name__ == '__main__':

    path_seed = './result_c100_{}_{}_{}_num_block={}_func_size={}_coupling_func={}'.format(args.method, args.network, args.func, args.num_block, args.func_size, args.coupling_func)
    if not os.path.isdir(path_seed):
        os.makedirs(path_seed)
    makedirs(path_seed)
    logger = get_logger(logpath=os.path.join(path_seed, 'logs_{}'.format(args.seed)), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
    
    if args.func == 'odetem':
        feature_layers = [ODEBlock(64)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    if args.func == 'coupling':
        feature_layers = [ODEBlock_coupling(64)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    if args.func == 'coupling2':
        feature_layers = [ODEBlock_coupling2(64)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    if args.func == 'temfunc':
        feature_layers = [ODEBlock_tem_func(64)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 100)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    x = torch.randn(1, 3, 28, 28).to(device)
    flops, params = profile(model, inputs=(x,))
    print('flops  of ODE is %.2fG' % (flops/1e9))
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_cifar100_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[30, 50, 70],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    inference_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_time_meter.update(time.time() - end)
        end = time.time()

        # if itr % batches_per_epoch == 0:
        #     with torch.no_grad():
        #         train_acc1, train_acc5 = accuracy(model, train_loader)
        #         inference_end = time.time()
        #         val_acc1, val_acc5 = accuracy(model, test_loader)
        #         inference_time_meter.update(time.time() - inference_end)
        #         if val_acc1 > best_acc:
        #             torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(path_seed, 'model_{}.pth'.format(args.seed)))
        #             best_acc = val_acc1
        #         logger.info(
        #             "Epoch {:04d} | " "Train_1 Acc {:.4f} | Train_5 Acc {:.4f} | Test_1 Acc {:.4f} | Test_5 Acc {:.4f} |Train Time {:.3f} ({:.3f}) |Test Time {:.3f} ({:.3f})".format(
        #                 itr // batches_per_epoch, train_acc1, train_acc5, val_acc1, val_acc5, batch_time_meter.val, batch_time_meter.avg, inference_time_meter.val, inference_time_meter.avg
        #             )
        #         )
        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                
                train_acc = accuracy_all(model, train_loader)
                inference_end = time.time()
                val_acc = accuracy_all(model, test_loader)
                inference_time_meter.update(time.time() - inference_end)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(path_seed, 'model_{}.pth'.format(args.seed)))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | " "Train Acc {:.4f} |Test Acc {:.4f} |Test Time {:.3f} ({:.3f})".format(
                        itr // batches_per_epoch, train_acc, val_acc, inference_time_meter.val, inference_time_meter.avg
                    )
                )
