import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def build_model(self, net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, warp1_yuan, warp2_yuan, a, canvas):
    a = 1/a
    out = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    learned_mask1 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out
    learned_mask2 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out)

    # (optional) draw composition images with different colors like our paper
    s1 = ((warp1_tensor[0] + 1) * 127.5 * learned_mask1[0]).cpu().detach().numpy().transpose(1, 2, 0)
    s2 = ((warp2_tensor[0] + 1) * 127.5 * learned_mask2[0]).cpu().detach().numpy().transpose(1, 2, 0)
    fusion = np.zeros((warp1_tensor.shape[2], warp1_tensor.shape[3], 3), np.uint8)
    fusion[..., 0] = s2[..., 0]
    fusion[..., 1] = s1[..., 1] * 0.5 + s2[..., 1] * 0.5
    fusion[..., 2] = s1[..., 2]
    path = self.datapath + "composition_color.jpg"
    cv2.imwrite(path, fusion)

    stitched_image_little = (warp1_tensor+1.) * learned_mask1 + (warp2_tensor+1.)*learned_mask2 - 1.
    stitched_image_little = ((stitched_image_little[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    learned_mask1 = (learned_mask1[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)
    learned_mask2 = (learned_mask2[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)

    if not self.LIIF:

        pil_image1 = Image.fromarray(np.uint8(learned_mask1))
        pil_image2 = Image.fromarray(np.uint8(learned_mask2))

        target_size1 = (warp1_yuan.shape[2], warp1_yuan.shape[1])
        target_size2 = (warp2_yuan.shape[2], warp2_yuan.shape[1])

        enlarged_mask1 = pil_image1.resize(target_size1, resample=Image.BICUBIC)
        enlarged_mask2 = pil_image2.resize(target_size2, resample=Image.BICUBIC)
    else:
        import liif_api
        LIIF_out_path1 = self.datapath + "LIIF_1.jpg"
        LIIF_out_path2 = self.datapath + "LIIF_2.jpg"
        enlarged_mask1 = liif_api.LIIF(learned_mask1, int(warp1_yuan.shape[1]), int(warp1_yuan.shape[2]), self.LIIF_model,
                                       LIIF_out_path1,self.gpu)
        enlarged_mask2 = liif_api.LIIF(learned_mask2, int(warp2_yuan.shape[1]), int(warp2_yuan.shape[2]), self.LIIF_model,
                                       LIIF_out_path2, self.gpu)

    enlarged_mask1 = np.clip(enlarged_mask1, 0, 255).astype(np.uint8)
    enlarged_mask2 = np.clip(enlarged_mask2, 0, 255).astype(np.uint8)

    if self.ract_mask:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        enlarged_mask1 = cv2.bitwise_and(enlarged_mask1, canvas)
        enlarged_mask2 = cv2.bitwise_and(enlarged_mask2, canvas)

    enlarged_mask1 = enlarged_mask1.astype(dtype=np.float32)
    enlarged_mask1 = (enlarged_mask1 / 255)
    enlarged_mask1 = np.transpose(enlarged_mask1, [2, 0, 1])
    enlarged_mask2 = enlarged_mask2.astype(dtype=np.float32)
    enlarged_mask2 = (enlarged_mask2 / 255)
    enlarged_mask2 = np.transpose(enlarged_mask2, [2, 0, 1])
    stitched_image = (warp1_yuan+1.) * enlarged_mask1 + (warp2_yuan+1.)*enlarged_mask2 - 1.
    stitched_image = ((stitched_image + 1.) * 127.5).transpose(1, 2, 0)
    enlarged_mask1 = (enlarged_mask1 * 255).transpose(1, 2, 0)
    enlarged_mask2 = (enlarged_mask2 * 255).transpose(1, 2, 0)

    out_dict = {}
    out_dict.update(learned_mask1=learned_mask1, learned_mask2=learned_mask2, stitched_image_little = stitched_image_little, stitched_image = stitched_image,
                    enlarged_mask1=enlarged_mask1, enlarged_mask2=enlarged_mask2)
    return out_dict
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src
### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

# predict the composition mask of img1
class Network(nn.Module):
    def __init__(self, nclasses=1):
        super(Network, self).__init__()

        self.stage1 = RSU7(3, 32, 64)
        self.pool12 = nn.MaxPool2d(2,stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # deoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)


        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1),
            nn.Sigmoid()
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, y, m1, m2):
        hx = x
        hy = y

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hy1 = self.stage1(hy)
        hy = self.pool12(hy1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hy2 = self.stage2(hy)
        hy = self.pool23(hy2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hy3 = self.stage3(hy)
        hy = self.pool34(hy3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hy4 = self.stage4(hy)
        hy = self.pool45(hy4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hy5 = self.stage5(hy)
        hy = self.pool56(hy5)

        hx6 = self.stage6(hx)
        # hx6up = _upsample_like(hx6, hx5)
        hy6 = self.stage6(hy)
        h6up = _upsample_like(hx6-hy6, hx5-hy5)

        #---------decoder----------#
        hx5d = self.stage5d(torch.cat((h6up, hx5-hy5), 1))
        h5up = _upsample_like(hx5d, hx4-hy4)

        hx4d = self.stage4d(torch.cat((h5up, hx4-hy4), 1))
        h4up = _upsample_like(hx4d, hx3-hy3)

        hx3d = self.stage3d(torch.cat((h4up, hx3-hy3), 1))
        h3up = _upsample_like(hx3d, hx2-hy2)

        hx2d = self.stage2d(torch.cat((h3up, hx2-hy2), 1))
        h2up = _upsample_like(hx2d, hx1-hy1)

        hx1d = self.stage1d(torch.cat((h2up, hx1-hy1), 1))

        d0 = self.out(hx1d)

        return d0






