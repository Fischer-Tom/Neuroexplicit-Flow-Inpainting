import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionBlock(nn.Module):
    def __init__(self, c, tau, grads, **kwargs):
        super().__init__()

        self.pad = nn.ReplicationPad2d(1)
        grad_x1, grad_x2, grad_y1, grad_y2 = self.get_weight(c)
        self.grad_x1 = nn.Parameter(grad_x1, requires_grad=False)
        self.grad_x2 = nn.Parameter(grad_x2, requires_grad=False)
        self.grad_y1 = nn.Parameter(grad_y1, requires_grad=False)
        self.grad_y2 = nn.Parameter(grad_y2, requires_grad=False)
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=True)

    def forward(self, u, a, b, c, alpha, beta):
        """w1a = w1a[:,:,:-1,:-1]
        w2a = w2a[:,:,:-1,:-1]
        w1b = w1b[:,:,:-1,:-1]
        w2b = w2b[:,:,:-1,:-1]
        w1c = w1c[:,:,:-1,:-1]
        w2c = w2c[:,:,:-1,:-1]
        """
        ux1 = F.conv2d(self.pad(u), self.grad_x1, groups=u.size(1))
        ux2 = F.conv2d(self.pad(u), self.grad_x2, groups=u.size(1))
        uy1 = F.conv2d(self.pad(u), self.grad_y1, groups=u.size(1))
        uy2 = F.conv2d(self.pad(u), self.grad_y2, groups=u.size(1))

        u1, u2, u3, u4 = Diff_Tensor(a, b, c, alpha, beta, ux1, ux2, uy1, uy2)

        uxx1 = F.conv2d(u1, -self.grad_x1.flip(2).flip(3), groups=ux1.size(1))
        uxx2 = F.conv2d(u2, -self.grad_x2.flip(2).flip(3), groups=ux2.size(1))

        uyy1 = F.conv2d(u3, -self.grad_y1.flip(2).flip(3), groups=uy1.size(1))
        uyy2 = F.conv2d(u4, -self.grad_y2.flip(2).flip(3), groups=uy2.size(1))

        Au = uxx1 + uxx2 + uyy1 + uyy2
        u = u + self.tau * Au

        return u

    def get_weight(self, c, h1=1, h2=1):
        hx = 1 / (h1)
        hy = 1 / (h2)
        weightx1 = torch.zeros((1, 1, 2, 2))
        weightx2 = torch.zeros((1, 1, 2, 2))
        weighty1 = torch.zeros((1, 1, 2, 2))
        weighty2 = torch.zeros((1, 1, 2, 2))
        weightx1[0][0][0][0] = -hx
        weightx1[0][0][0][1] = hx
        weightx2[0][0][1][0] = -hx
        weightx2[0][0][1][1] = hx
        weighty1[0][0][0][0] = -hy
        weighty1[0][0][1][0] = hy
        weighty2[0][0][0][1] = -hy
        weighty2[0][0][1][1] = hy
        image_weight_x1 = weightx1.repeat(c, 1, 1, 1)
        image_weight_x2 = weightx2.repeat(c, 1, 1, 1)
        image_weight_y1 = weighty1.repeat(c, 1, 1, 1)
        image_weight_y2 = weighty2.repeat(c, 1, 1, 1)

        return image_weight_x1, image_weight_x2, image_weight_y1, image_weight_y2


@torch.jit.script
def Diff_Tensor(a, b, c, alpha, beta, ux1, ux2, uy1, uy2):
    # sign = torch.sign(b)
    # beta = (1. - 2. * alpha) * sign

    w1a = a * (1 - alpha) / 2
    w2a = a * alpha / 2
    w1b = b * (1 - beta) / 4
    w2b = b * (1 + beta) / 4
    w1c = c * (1 - alpha) / 2
    w2c = c * alpha / 2

    u1 = w1a * ux1 + w2a * ux2 + w1b * uy1 + w2b * uy2
    u2 = w2a * ux1 + w1a * ux2 + w2b * uy1 + w1b * uy2
    u3 = w1b * ux1 + w2b * ux2 + w1c * uy1 + w2c * uy2
    u4 = w2b * ux1 + w1b * ux2 + w2c * uy1 + w1c * uy2

    return u1, u2, u3, u4


class WWWDiffusion(nn.Module):
    def __init__(self, tau, grads, **kwargs):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=True)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, u, a, b, c, alpha, beta):
        tau = 1 / (4 * (1 - alpha.min()))
        Au = get_delta(self.pad(u), a, b, c, alpha, beta)

        u = u + tau * Au

        return u


def get_delta(u, a, b, c, alpha, beta):
    _, _, w, h = u.shape
    h = h - 1
    w = w - 1

    sign = torch.sign(b)
    beta = (1.0 - 2.0 * alpha) * sign
    delta = alpha * (a + c) + beta * b
    wpo = 0.5 * (
        a[:, :, 1:w, 1:h]
        - delta[:, :, 1:w, 1:h]
        + a[:, :, 1:w, 0 : h - 1]
        - delta[:, :, 1:w, 0 : h - 1]
    )
    wmo = 0.5 * (
        a[:, :, 0 : w - 1, 1:h]
        - delta[:, :, 0 : w - 1, 1:h]
        + a[:, :, 0 : w - 1, 0 : h - 1]
        - delta[:, :, 0 : w - 1, 0 : h - 1]
    )
    wop = 0.5 * (
        c[:, :, 1:w, 1:h]
        - delta[:, :, 1:w, 1:h]
        + c[:, :, 0 : w - 1, 1:h]
        - delta[:, :, 0 : w - 1, 1:h]
    )
    wom = 0.5 * (
        c[:, :, 1:w, 0 : h - 1]
        - delta[:, :, 1:w, 0 : h - 1]
        + c[:, :, 0 : w - 1, 0 : h - 1]
        - delta[:, :, 0 : w - 1, 0 : h - 1]
    )
    wpp = 0.5 * (b[:, :, 1:w, 1:h] + delta[:, :, 1:w, 1:h])
    wmm = 0.5 * (b[:, :, 0 : w - 1, 0 : h - 1] + delta[:, :, 0 : w - 1, 0 : h - 1])
    wmp = 0.5 * (delta[:, :, 0 : w - 1, 1:h] - b[:, :, 0 : w - 1, 1:h])
    wpm = 0.5 * (delta[:, :, 1:w, 0 : h - 1] - b[:, :, 1:w, 0 : h - 1])
    woo = -wpo - wmo - wop - wom - wpp - wmm - wmp - wpm
    Au = (
        woo * u[:, :, 1:w, 1:h]
        + wpo * u[:, :, 2:, 1:h]
        + wmo * u[:, :, 0 : w - 1, 1:h]
        + wop * u[:, :, 1:w, 2:]
        + wom * u[:, :, 1:w, 0 : h - 1]
        + wpp * u[:, :, 2:, 2:]
        + wmm * u[:, :, 0 : w - 1, 0 : h - 1]
        + wpm * u[:, :, 2:, 0 : h - 1]
        + wmp * u[:, :, 0 : w - 1, 2:]
    )

    return Au


class PeronaMalikDiffusivity(nn.Module):
    def __init__(self, contrast=1.0):
        super().__init__()
        self.contrast = nn.Parameter(torch.tensor(contrast), requires_grad=True)

    def forward(self, x):
        # Adapted to enforce contrast parameter >0
        divisor = (x * x) / (self.contrast * self.contrast + 1e-8)

        return 1 / (1 + divisor)


class CharbonnierDiffusivity(nn.Module):
    def __init__(self, contrast=1.0):
        super().__init__()
        self.contrast = nn.Parameter(torch.tensor(contrast), requires_grad=True)

    def forward(self, x):
        # Adapted to enforce contrast parameter >0
        divisor = (x * x) / (self.contrast * self.contrast + 1e-8)

        return 1 / torch.sqrt((1 + divisor))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        ks,
        stride=1,
        pad=1,
        pad_mode="zeros",
        bias=True,
        act=nn.ReLU(),
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            padding_mode=pad_mode,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            padding_mode=pad_mode,
            bias=bias,
        )
        self.act = act

    def forward(self, x):
        identity = x
        x = self.conv2(self.act(self.conv1(x)))
        return self.act(identity + x)


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        ks,
        stride=1,
        pad=1,
        pad_mode="zeros",
        bias=True,
        act=nn.ReLU(),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            padding_mode=pad_mode,
            bias=bias,
        )
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class SimpleUpConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        ks,
        stride=2,
        pad=1,
        output_padding=0,
        bias=True,
        act=nn.ReLU(),
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            output_padding=output_padding,
            bias=bias,
        )
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class DiffusivityModule_Deep(nn.Module):
    def __init__(self, dim, learned_mode):
        super().__init__()
        self.conv0 = SimpleConv(3, dim, 3, 1, 1)
        self.conv1 = SimpleConv(dim, dim, 3, 2, 1)
        self.conv1_1 = SimpleConv(dim, dim, 3, 1, 1)
        self.conv2 = SimpleConv(dim, dim * 2, 3, 2, 1)
        self.conv2_1 = SimpleConv(dim * 2, dim * 2, 3, 1, 1)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 3, 2, 1)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 4, 3, 2, 1)
        self.conv4_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv5 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)

        self.deconv5 = SimpleUpConv(dim * 8, dim * 4, 1, 2, 0, 1)
        self.deconv4 = SimpleUpConv(dim * 4 + dim * 4, dim * 4, 1, 2, 0, 1)
        self.deconv3 = SimpleUpConv(dim * 4 + dim * 4, dim * 4, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim * 4 + dim * 2, dim * 2, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim * 2 + dim * 1, dim, 1, 2, 0, 1)

        self.DT0 = nn.Conv2d(
            in_channels=dim + dim, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.DT1 = nn.Conv2d(
            in_channels=dim * 2 + dim,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT2 = nn.Conv2d(
            in_channels=dim * 4 + dim * 2,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT3 = nn.Conv2d(
            in_channels=dim * 4 + dim * 4,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT4 = nn.Conv2d(
            in_channels=dim * 4 + dim * 4,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu = nn.ReLU()

    def forward(self, I):
        x0 = self.conv0(I)
        x1 = self.conv1_1(self.conv1(x0))
        x2 = self.conv2_1(self.conv2(x1))
        x3 = self.conv3_1(self.conv3(x2))
        x4 = self.conv4_1(self.conv4(x3))
        x5 = self.conv5(x4)

        x4 = torch.cat((self.deconv5(x5), x4), dim=1)
        x3 = torch.cat((self.deconv4(x4), x3), dim=1)
        x2 = torch.cat((self.deconv3(x3), x2), dim=1)
        x1 = torch.cat((self.deconv2(x2), x1), dim=1)
        x0 = torch.cat((self.deconv1(x1), x0), dim=1)

        x4 = self.DT4(x4)
        x3 = self.DT3(x3)
        x2 = self.DT2(x2)
        x1 = self.DT1(x1)
        x0 = self.DT0(x0)

        return [x4, x3, x2, x1, x0]


class DiffusivityModule(nn.Module):
    def __init__(self, dim: int, learned_mode=5):
        super().__init__()
        self.conv0 = SimpleConv(3, dim, 3, 1, 1)
        self.conv1 = SimpleConv(dim, dim, 3, 2, 1)
        self.conv1_1 = SimpleConv(dim, dim, 3, 1, 1)
        self.conv2 = SimpleConv(dim, dim * 2, 3, 2, 1)
        self.conv2_1 = SimpleConv(dim * 2, dim * 2, 3, 1, 1)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 3, 2, 1)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)

        self.deconv4 = SimpleUpConv(dim * 8, dim * 4, 1, 2, 0, 1)
        self.deconv3 = SimpleUpConv(dim * 4 + dim * 4, dim * 4, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim * 4 + dim * 2, dim * 2, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim * 2 + dim * 1, dim, 1, 2, 0, 1)

        self.DT0 = nn.Conv2d(
            in_channels=dim + dim,
            out_channels=learned_mode,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT1 = nn.Conv2d(
            in_channels=dim * 2 + dim,
            out_channels=learned_mode,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT2 = nn.Conv2d(
            in_channels=dim * 4 + dim * 2,
            out_channels=learned_mode,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DT3 = nn.Conv2d(
            in_channels=dim * 4 + dim * 4,
            out_channels=learned_mode,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu = nn.ReLU()

    def forward(self, I):
        b, _, w, h = I.shape
        x0 = self.conv0(I)
        x1 = self.conv1_1(self.conv1(x0))
        x2 = self.conv2_1(self.conv2(x1))
        x3 = self.conv3_1(self.conv3(x2))
        x4 = self.conv4(x3)

        x3 = torch.cat((self.deconv4(x4), x3), dim=1)
        x2 = torch.cat((self.deconv3(x3), x2), dim=1)
        x1 = torch.cat((self.deconv2(x2), x1), dim=1)
        x0 = torch.cat((self.deconv1(x1), x0), dim=1)

        x3 = self.DT3(x3)
        x2 = self.DT2(x2)
        x1 = self.DT1(x1)
        x0 = self.DT0(x0)

        return [x3, x2, x1, x0]

    def adjust_size(self, x):
        _, _, w, h = x.shape

        return F.interpolate(
            x, size=(w + (w % 2), h + (h % 2)), mode="bilinear", align_corners=True
        )
