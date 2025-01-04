from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data
import torch
import cv2


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NLBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True):
        """
        Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        参考: https://arxiv.org/abs/1711.07971
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: 2 (spatial)
            bn_layer: whether to add batch norm
        """
        super(NLBlock, self).__init__()
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    nn.BatchNorm2d(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (N, C, HW)
        g_x = g_x.permute(0, 2, 1)  # (N, HW, C)

        # pairwise functions（计算注意力权重）
        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)    # (N, C, HW)
            phi_x = x.view(batch_size, self.in_channels, -1)    # (N, C, HW)
            theta_x = theta_x.permute(0, 2, 1)  # (N, HW, C)
            f = torch.matmul(theta_x, phi_x)  # (N, HW, HW)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)    # (N, C, HW)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)    # (N, C, HW)
            theta_x = theta_x.permute(0, 2, 1)  # (N, HW, C)
            f = torch.matmul(theta_x, phi_x)  # (N, HW, HW)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)    # (N, C, HW, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)    # (N, C, 1, HW)

            h = theta_x.size(2)   # HW
            w = phi_x.size(3)   # HW
            theta_x = theta_x.repeat(1, 1, 1, w)    # (N, C, HW, HW)
            phi_x = phi_x.repeat(1, 1, h, 1)    # (N, C, HW, HW)

            concat = torch.cat([theta_x, phi_x], dim=1)    # (N, 2C, HW, HW)
            f = self.W_f(concat)    # (N, 1, HW, HW)
            f = f.view(f.size(0), f.size(2), f.size(3))    # (N, HW, HW)
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)    # f_div_C: (N, HW, HW), g_x: (N, HW, C), y: (N, HW, C)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # (N, C, HW)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (N, C, H, W)
        
        # residual connection
        z = self.W_z(y) + x

        return z


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W_x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=3, stride=2, padding=1, bias=True)
        self.W_g = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0, bias=True)
        self.W_f = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                            kernel_size=1, stride=1, padding=0, bias=True)

        self.W_y = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )

    def forward(self, x, g):
        '''
        :param x: (N, C, H, W)
        :param g: (N, 2C, H/2, W/2)
        :return: out

        key: 键
        query: 查询
        att_weight: 注意力权重
        '''

        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # W_x => (N, C, H, W) -> (N, C_inter, H/2, W/2)
        # W_g => (N, 2C, H/2, W/2) -> (N, C_inter, H/2, W/2)
        key = self.W_x(x)    # self.W_x是步长为2的3×3Conv，输入通道数为self.in_channels，输出通道数为self.inter_channels
        query = self.W_g(g)    # self.W_g是1×1Conv，输入通道数为self.gating_channels，输出通道数为self.inter_channels

        f = F.relu(key + query, inplace=True)
        # W_f => (N, 1, H/2, W/2)
        att_weight = F.sigmoid(self.W_f(f))   # self.W_f是1×1Conv，输入通道数为self.inter_channels，输出通道数为1

        # upsample the attentions and multiply
        att_weight = F.upsample(att_weight, size=input_size[2:], mode='bilinear')  # (N, 1, H/2, W/2) -> (N, 1, H, W)
        y = att_weight.expand_as(x) * x     # y = (N, C, H, W)

        # W_y => (N, C, H, W) -> (N, C, H, W)
        out = self.W_y(y)   # self.W是1×1Conv＋BN，输入和输出通道数都为self.in_channels

        return out


class Improved_ConvGRUCell(nn.Module):

    def __init__(self, input_channels, hidden_channels=None, kernel_size=3):
        super(Improved_ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels

        if hidden_channels is None:
            hidden_channels = input_channels // 2
            if hidden_channels == 0:
                hidden_channels = 1
        self.hidden_channels = hidden_channels

        self.reset_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        
        # for candidate neural memory
        self.out_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1, padding=0, bias=False)

        # init.orthogonal(self.reset_gate.weight)
        # init.orthogonal(self.update_gate.weight)
        # init.orthogonal(self.out_gate.weight)
        # init.constant(self.reset_gate.bias, 0.)
        # init.constant(self.update_gate.bias, 0.)
        # init.constant(self.out_gate.bias, 0.)


    def forward(self, x, h_prev=None, gamma_update=1.0):
        """
        x: (n, c, h, w)
        h_prev: (n, c_hidden, h, w)
        """
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty h_prev, if None is provided
        if h_prev is None:
            h_size = [batch_size, self.hidden_channels] + list(spatial_size)
            h_prev = Variable(torch.zeros(h_size)).to(device='cuda' if torch.cuda.is_available() else 'cpu')

        # data size is [batch, channel, height, width]
        combined = torch.cat([x, h_prev], dim=1)

        update = F.sigmoid(self.update_gate(combined) * gamma_update)
        reset = F.sigmoid(self.reset_gate(combined))
        h_tmp = F.tanh(self.out_gate(torch.cat([x, h_prev * reset], dim=1)))  # candidate
        h_cur = h_prev * update + h_tmp * (1 - update)

        out = self.conv(h_cur)

        return out, h_cur


class SSLSE(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 base_c=32):
        super(SSLSE, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels, base_c)
        self.Conv2 = nn.Sequential(*[ResBlock(base_c, base_c * 2, 1), ResBlock(base_c * 2, base_c * 2, 1)])
        self.Conv3 = nn.Sequential(*[ResBlock(base_c * 2, base_c * 4, 1), NLBlock(in_channels=base_c * 4)])
        self.Conv4 = nn.Sequential(*[ResBlock(base_c * 4, base_c * 8, 1), NLBlock(in_channels=base_c * 8)])
        self.Conv5 = nn.Sequential(*[ResBlock(base_c * 8, base_c * 16, 1), ResBlock(base_c * 16, base_c * 16, 1)])

        self.Up4 = up_conv(base_c * 16, base_c * 8)
        self.Att4 = AttentionGate(base_c * 8, base_c * 16)
        self.up_conv4 = conv_block(base_c * 16, base_c * 8)

        self.Up3 = up_conv(base_c * 8, base_c * 4)
        self.Att3 = AttentionGate(base_c * 4, base_c * 8)
        self.up_conv3 = conv_block(base_c * 8, base_c * 4)

        self.Up2 = up_conv(base_c * 4, base_c * 2)
        self.Att2 = AttentionGate(base_c * 2, base_c * 4)
        self.up_conv2 = conv_block(base_c * 4, base_c * 2)

        self.Up1 = up_conv(base_c * 2, base_c)
        self.Att1 = AttentionGate(base_c, base_c * 2)
        self.up_conv1 = conv_block(base_c * 2, base_c)

        self.Conv = nn.Conv2d(base_c, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        x4 = self.Att4(x=e4, g=e5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(x=e3, g=d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(x=e2, g=d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(x=e1, g=d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.up_conv1(d1)

        out = self.Conv(d1)

        return out


class VSLSE(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 base_c=32):
        super(VSLSE, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels, base_c)
        self.Conv2 = nn.Sequential(*[ResBlock(base_c, base_c * 2, 1), ResBlock(base_c * 2, base_c * 2, 1)])
        self.Conv3 = nn.Sequential(*[ResBlock(base_c * 2, base_c * 4, 1), NLBlock(in_channels=base_c * 4)])
        self.Conv4 = nn.Sequential(*[ResBlock(base_c * 4, base_c * 8, 1), NLBlock(in_channels=base_c * 8)])
        self.Conv5 = nn.Sequential(*[ResBlock(base_c * 8, base_c * 16, 1), ResBlock(base_c * 16, base_c * 16, 1)])

        self.Gru1 = Improved_ConvGRUCell(input_channels=base_c)
        self.Gru2 = Improved_ConvGRUCell(input_channels=base_c * 2)
        self.Gru3 = Improved_ConvGRUCell(input_channels=base_c * 4)
        self.Gru4 = Improved_ConvGRUCell(input_channels=base_c * 8)
        self.Gru5 = Improved_ConvGRUCell(input_channels=base_c * 16)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.hidden4 = None
        self.hidden5 = None

        self.Up4 = up_conv(base_c * 16, base_c * 8)
        self.Att4 = AttentionGate(base_c * 8, base_c * 16)
        self.up_conv4 = conv_block(base_c * 16, base_c * 8)

        self.Up3 = up_conv(base_c * 8, base_c * 4)
        self.Att3 = AttentionGate(base_c * 4, base_c * 8)
        self.up_conv3 = conv_block(base_c * 8, base_c * 4)

        self.Up2 = up_conv(base_c * 4, base_c * 2)
        self.Att2 = AttentionGate(base_c * 2, base_c * 4)
        self.up_conv2 = conv_block(base_c * 4, base_c * 2)

        self.Up1 = up_conv(base_c * 2, base_c)
        self.Att1 = AttentionGate(base_c, base_c * 2)
        self.up_conv1 = conv_block(base_c * 2, base_c)

        self.Conv = nn.Conv2d(base_c, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, gamma_update=1.0):

        e1 = self.Conv1(x)
        e1, self.hidden1 = self.Gru1(e1, self.hidden1, gamma_update)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2, self.hidden2 = self.Gru2(e2, self.hidden2, gamma_update)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3, self.hidden3 = self.Gru3(e3, self.hidden3, gamma_update)
        

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4, self.hidden4 = self.Gru4(e4, self.hidden4, gamma_update)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5, self.hidden5 = self.Gru5(e5, self.hidden5, gamma_update)

        d4 = self.Up4(e5)
        x4 = self.Att4(x=e4, g=e5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(x=e3, g=d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(x=e2, g=d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(x=e1, g=d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.up_conv1(d1)

        out = self.Conv(d1)

        return out
    
    def infer(self, prev_mask, cur_img, t):

        # 清空隐藏状态
        if t == 0:
            self.init_hidden_state()

        ########### 计算控制系数 ###########
        #默认
        gamma_update = 1.0

        _, th = cv2.threshold(cur_img, 200, 1, cv2.THRESH_BINARY)
        N_total = th.sum()

        N_target = (prev_mask == prev_mask.max()).sum()

        delta = N_total - N_target
        if delta < 10000:
            gamma_update = 1.0
        elif delta > 40000:
            gamma_update = 2.0
        else:
            gamma_update = delta / 30000 + 2 / 3

        return self.forward(cur_img, gamma_update)
    
    def init_hidden_state(self):
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.hidden4 = None
        self.hidden5 = None


if __name__ == '__main__':

    # # test ResBlock
    # print('ResBlock:')
    # x = torch.zeros(1, 3, 28, 28)
    # for out_channels in [3, 10]:
    #     for stride in [1, 2]:
    #         print('out_channels: {}, stride: {}'.format(out_channels, stride))
    #         block = ResBlock(3, out_channels, stride)
    #         y = block(x)
    #         print(y.shape)

    # # test NLBlock
    # print('NLBlock:')
    # for bn_layer in [True, False]:
    #     print('bn_layer = True') if bn_layer else print('bn_layer = False')
    #     img = torch.zeros(2, 3, 20, 20)
    #     net = NLBlock(in_channels=3, mode='concatenate', bn_layer=bn_layer)
    #     out = net(img)
    #     print(out.size())

    # # test AttentionGate
    # model = AttentionGate(3, 16)
    # x = torch.randn((1, 3, 288, 288))
    # g = torch.randn((1, 16, 144, 144))
    # y = model(x, g)
    # print(y.shape)
    # pass

    # test Improved_ConvGRUCell
    t = 5
    x = torch.zeros(1, t, 3, 28, 28)
    block = Improved_ConvGRUCell(3)
    hidden = None
    for i in range(t):
        _, hidden = block(x[:, i, :, :], hidden)
        print(i, _.shape, hidden.shape)

    # # test SSLSE
    # model = SSLSE(1, 1, 32)
    # # model.load_state_dict(torch.load('exps/exp12/sslse_epoch_200_batchsize_8.pth', map_location=torch.device('cpu')))
    # x = torch.randn((1, 1, 288, 288))
    # y = model(x)
    # print(y.shape)


    # # test VSLSE
    # t = 10
    # x = torch.randn((1, t, 1, 288, 288))
    # hiddens = [None] * (t+1)
    # model = VSLSE(1, 1, 32)

    # # ckpt_pre = torch.load('exps/exp12/sslse_epoch_200_batchsize_8.pth', map_location=torch.device('cpu'))
    # # ckpt = model.state_dict()
    # # ckpt.update(ckpt_pre)
    # # model.load_state_dict(ckpt)

    # hiddens[0] = model.hidden2
    # for i in range(t):
    #     y = model(x[:, i, :, :])
    #     hiddens[i+1] = model.hidden2
    #     print(y.shape)
    # pass







    
    
    # import cv2
    # from PIL import Image
    # import torchvision
    # from loss import threshold_predictions_p

    # # img = cv2.imread('/zgh/dataset/WeldSeam/images/0747.jpg', 0)
    # img = Image.open('/zgh/dataset/WeldSeam/images/0747.jpg').convert('L')
    # data_transform = torchvision.transforms.Compose([
    #        torchvision.transforms.Resize((288,288)),
    #      #   torchvision.transforms.CenterCrop(96),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=[0.3], std=[0.32])
    #     ])
    # img = data_transform(img)

    # model.to(device='cuda')
    # img = img.unsqueeze(0).to(device='cuda')
    
    # with torch.no_grad():
    #     prediction = model(img)
    # prediction = F.sigmoid(prediction).cpu()
    # prediction = threshold_predictions_p(prediction, 0.9)
    # prediction = prediction.numpy()

    # cv2.imshow("", prediction[0][0])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
