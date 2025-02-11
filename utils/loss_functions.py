import torch
import torch.nn.functional as F


def Scaled_EPE_Loss_sum(pred_flow, target_flow):
    b, _, h, w = pred_flow.size()
    scaled_target = F.interpolate(target_flow, (h, w), mode="area")
    return torch.norm(scaled_target - pred_flow, 2, 1).sum() / b


def Scaled_EPE_Loss_mean(pred_flow, target_flow):
    b, _, h, w = pred_flow.size()
    scaled_target = F.interpolate(target_flow, (h, w), mode="area")
    return torch.norm(scaled_target - pred_flow, 2, 1).mean()


def MultiScale_EPE_Loss(model_output, target_flow, mode="mean"):
    loss = 0.0
    if mode == "sum":
        weights = [0.005, 0.01, 0.02, 0.08, 0.32, 0.64, 1.28]

        for out, weight in zip(model_output, weights):
            loss += weight * Scaled_EPE_Loss_sum(out, target_flow)
    else:
        weights = [0.32, 0.08, 0.04, 0.02, 0.01]
        for out, weight in zip(model_output, weights):
            loss += weight * Scaled_EPE_Loss_mean(out, target_flow)
    return loss


def RMSE_Loss(pred, target, occ, train=False):
    b, _, h, w = pred.shape
    return ((pred - target)[occ.bool()] ** 2).mean()


def EPE_Loss(pred_flow, target_flow, occ_flow=None, train=False):
    b, _, h, w = pred_flow.size()
    # scaled_target = F.interpolate(target_flow, (h,w), mode='bilinear')
    loss = torch.norm(target_flow - pred_flow, 2, 1)
    if occ_flow is None:
        return loss.mean(), loss.mean()
    epe = loss.mean()
    if train:
        return epe
    mag_gt = torch.sum(target_flow**2, dim=1).sqrt()

    epex3 = loss > 3.0
    ex = loss > 0.05 * mag_gt
    loss = epex3 * ex
    return loss.float().mean(), epe
    return loss[occ_flow.bool().squeeze(1)].mean()

def charbonnier(x, alpha=0.25, epsilon=1.e-9):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)
def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.mean(s_loss)
def EPE_Loss_Mask(pred_flow, target_flow, mask):
    loss = torch.norm(target_flow - pred_flow, 2, 1)
    epe = loss[mask.bool().squeeze(1)].mean()
    return epe


def EPE_Loss_Sum(pred_flow, target_flow):
    b, _, h, w = target_flow.size()
    return torch.norm(target_flow - pred_flow, 2, 1).sum()


def Eigen_Loss(I, v1):
    w1 = get_structure_tensor(I)

    n1 = torch.norm(v1 - w1, 2, 1).mean()

    return n1


def get_structure_tensor(I1):
    # TODO: Add (1-alpha) component
    _, _, h, w = I1.shape
    w = w + 1
    h = h + 1
    I1 = F.pad(I1, (1, 1, 1, 1), mode="replicate")
    Ixo = I1[:, :, 2:, 1:h] - I1[:, :, 1:w, 1:h]
    Ixp = I1[:, :, 2:, 2:] - I1[:, :, 1:w, 2:]
    Iyo = I1[:, :, 1:w, 2:] - I1[:, :, 1:w, 1:h]
    Iyp = I1[:, :, 2:, 2:] - I1[:, :, 2:, 1:h]

    Ixx = 0.5 * (Ixo * Ixo + Ixp * Ixp)
    Iyy = 0.5 * (Iyo * Iyo + Iyp * Iyp)
    Ixy = 0.25 * (Ixo + Ixp) * (Iyo + Iyp)

    v11, v12, mu1, mu2 = get_eigenvectors(Ixx, Iyy, Ixy)

    v1 = torch.cat((v11, v12), dim=1)
    return v1


def get_eigenvectors(Ixx, Iyy, Ixy):
    """
    Computes Eigenvectors of the Structure Tensor with the principal axis transformation
    :param ux2: squared x derivative
    :param uy2: squared y derivaitve
    :param uxy: x derivative x y derivative
    :return: Difussion tensor a,b,c
    """
    Ixx = torch.sum(Ixx, dim=1, keepdim=True)
    Iyy = torch.sum(Iyy, dim=1, keepdim=True)
    Ixy = torch.sum(Ixy, dim=1, keepdim=True)
    aux = torch.sqrt((Iyy - Ixx) * (Iyy - Ixx) + 4.0 * Ixy * Ixy)

    iso = aux == 0.0

    mu1 = 0.5 * (Ixx + Iyy + aux)
    mu2 = 0.5 * (Ixx + Iyy - aux)

    order = Ixx > Iyy

    v11 = torch.zeros_like(Ixx)
    v12 = torch.zeros_like(Ixx)

    temp1 = Ixx - Iyy + aux
    temp2 = 2.0 * Ixy

    v11[order] = temp1[order]
    v12[order] = temp2[order]

    temp1 = Iyy - Ixx + aux
    order = torch.logical_not(order)
    v12[order] = temp1[order]
    v11[order] = temp2[order]

    v11[iso] = 1.0
    v12[iso] = 0.0
    mu1[iso] = Ixx[iso]
    mu2[iso] = Ixx[iso]

    norm = torch.sqrt(v11 * v11 + v12 * v12)
    low_norm = norm < 1e-6

    v11[low_norm] = 1.0
    v12[low_norm] = 0.0

    high_norm = torch.logical_not(low_norm)
    v11[high_norm] = v11[high_norm] / norm[high_norm]
    v12[high_norm] = v12[high_norm] / norm[high_norm]

    return v11, v12, mu1, mu2
