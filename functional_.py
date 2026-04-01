import cv2
import numpy as np
import os
import time
import faiss
import shutil
from memory_profiler import profile
import argparse
import torch
from SRMN import build_model, Network
import glob
import pyceres


def out_point(self,D,I,kp1,kp2,img1,img2):

    if self._GMS:
        matches = []
        for i in range(D.shape[0]):
            if D[i][0] < D[i][1] * 0.7:
                matches.append(cv2.DMatch(i, I[i][0], 0, D[i][0]))
        matches_gms = cv2.xfeatures2d.matchGMS(
            img1.shape[:2], img2.shape[:2], kp1, kp2, matches,
            withScale=False, withRotation=False, thresholdFactor=6)
        # 使用cv2.findHomography计算单应性矩阵
        src_pts = []
        dst_pts = []
        for match in matches_gms:
            src_pts.append(kp1[match.queryIdx].pt)
            dst_pts.append(kp2[match.trainIdx].pt)
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    else:
        good = []
        for j in range(D.shape[0]):
            if D[j][0] < D[j][1] * 0.7:
                good.append(cv2.DMatch(j, I[j][0], 0, D[j][0]))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    return src_pts, dst_pts

def _rmse(trans_point,origin_point,img1,img2,r,total_rmse):
    trans_point = np.int32(trans_point)
    origin_point = np.int32(origin_point)
    trans_point = np.reshape(trans_point, (trans_point.shape[0], -1))
    origin_point = np.reshape(origin_point,(origin_point.shape[0], -1))

    for k in range(trans_point.shape[0]):
        p = img1[origin_point[k][1], origin_point[k][0]]
        q = img2[trans_point[k][1], trans_point[k][0]]
        r += (np.float32((p - q) ** 2)).sum()
    r = (r / (trans_point.shape[0] * 3)) ** 0.5
    total_rmse += r

    return r,total_rmse

def KP_transform(arr,img):
    arr = arr[:2, :]
    kp_sift = []

    n = arr.shape[1]

    for i in range(n):
        x = arr[0, i]
        y = arr[1, i]
        kp_sift.append(cv2.KeyPoint(x, y, 1))

    return kp_sift

def gray_color(grayim):
    height, width = grayim.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    rgb_image[:, :, 0] = grayim
    rgb_image[:, :, 1] = grayim
    rgb_image[:, :, 2] = grayim
    rgb_image *= 255
    rgb_image = rgb_image.astype(np.uint8)
    return rgb_image


class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1,  padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc

class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self,
                 weights_path,
                    nms_dist,
                    conf_thresh,
                    nn_thresh,
                    cuda,
                    device=1):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.device = device

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda(device=self.device)
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])  # 返回降序索引
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda(device=self.device)
        # Forward pass of network.
        outs = self.net.forward(inp)  # semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        # desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        semi, coarse_desc = outs[0], outs[1]  # channel:256
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove  # 靠近边界移除  4
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda(device=self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


def read_image(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
    impath: Path to input image.
    img_size: (W, H) tuple specifying resize size.
    Returns
    grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

def frame2tensor(frame, device):
    return torch.from_numpy(frame).float()[None, None].to(device)

def draw_img_keypoint(self,keypoints,img,i,param):
    kp_cv2 = []

    for point in keypoints:
        x, y = point
        kp = cv2.KeyPoint(x, y, 1)
        kp_cv2.append(kp)

    img_with_keypoints = np.copy(img)
    for kp in kp_cv2:
        x, y = kp.pt
        size = kp.size

        cv2.circle(img_with_keypoints, (int(x), int(y)), 10, (0, 0, 255), -1)
    self.point_loacation_file = self.log_path + "/point_location/" +param
    if not os.path.exists(self.point_loacation_file):
        os.makedirs(self.point_loacation_file)
    output_path = os.path.join(self.point_loacation_file, self.filename[i].split('.')[0][-4:] + ".JPG")
    cv2.imwrite(output_path, img_with_keypoints)
    return kp_cv2, img_with_keypoints

def draw_img_match(self,src_img, dst_img, src_pts, dst_pts, i):
    matches = []
    for q in range(len(src_pts)):
        match = cv2.DMatch(q, q, 0, 0)
        matches.append(match)

    matched_img = cv2.drawMatches(src_img, src_pts, dst_img, dst_pts, matches,outImg=None, matchColor=(0, 0, 255),flags=cv2.LINE_AA)

    output_path = os.path.join(self.point_loacation_file,"match_" + self.filename[i].split('.')[0][-4:] + '_' + self.filename[i + 1].split('.')[0][-4:] + ".JPG")
    cv2.imwrite(output_path, matched_img)

def get_mask_img(filename, i, log_path,sizer1,sizer2, H_translation, H, x_max, x_min, y_max, y_min, img1_, img2_):
    white_image1 = np.zeros((sizer1[0], sizer1[1], 3), np.uint8)
    white_image1.fill(255)
    white_image2 = np.zeros((sizer2[0], sizer2[1], 3), np.uint8)
    white_image2.fill(255)
    mask1_warp = cv2.warpPerspective(white_image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    mask2_warp = cv2.warpPerspective(white_image2, H_translation.astype(np.float32), (x_max - x_min, y_max - y_min))
    warp_image1 = cv2.warpPerspective(img1_, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    warp_image2 = cv2.warpPerspective(img2_, H_translation.astype(np.float32), (x_max - x_min, y_max - y_min))
    mask_img_path = log_path
    end_name = filename[i].split('.')[0][-4:] + '_' + filename[i + 1].split('.')[0][-4:] + ".JPG"
    # clear_folder(mask_img_path)
    warp1_file = mask_img_path + "/warp_mask/warp1/"
    warp2_file = mask_img_path + "/warp_mask/warp2/"
    mask1_file = mask_img_path + "/warp_mask/mask1/"
    mask2_file = mask_img_path + "/warp_mask/mask2/"
    if not os.path.exists(warp1_file):
        os.makedirs(warp1_file)
    if not os.path.exists(warp2_file):
        os.makedirs(warp2_file)
    if not os.path.exists(mask1_file):
        os.makedirs(mask1_file)
    if not os.path.exists(mask2_file):
        os.makedirs(mask2_file)
    cv2.imwrite(warp1_file + end_name, warp_image1)
    cv2.imwrite(warp2_file + end_name, warp_image2)
    cv2.imwrite(mask1_file + end_name, mask1_warp)
    cv2.imwrite(mask2_file + end_name, mask2_warp)
# @profile
def get_mask_img_UDIS(filename, i, log_path, sizer1,sizer2, H_translation, H, x_max, x_min, y_max, y_min, img1_, img2_):
    white_image1 = np.zeros((sizer1[0], sizer1[1], 3), np.uint8)
    white_image1.fill(255)
    white_image2 = np.zeros((sizer2[0], sizer2[1], 3), np.uint8)
    white_image2.fill(255)
    mask1_warp = cv2.warpPerspective(white_image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    mask2_warp = cv2.warpPerspective(white_image2, H_translation.astype(np.float32), (x_max - x_min, y_max - y_min))
    warp_image1 = cv2.warpPerspective(img1_, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    warp_image2 = cv2.warpPerspective(img2_, H_translation.astype(np.float32), (x_max - x_min, y_max - y_min))
    mask_img_path = log_path
    end_name = filename[i].split('.')[0][-6:]+ ".JPG"
    # clear_folder(mask_img_path)
    warp1_file = mask_img_path + "/warp_mask/warp1/"
    warp2_file = mask_img_path + "/warp_mask/warp2/"
    mask1_file = mask_img_path + "/warp_mask/mask1/"
    mask2_file = mask_img_path + "/warp_mask/mask2/"
    if not os.path.exists(warp1_file):
        os.makedirs(warp1_file)
    if not os.path.exists(warp2_file):
        os.makedirs(warp2_file)
    if not os.path.exists(mask1_file):
        os.makedirs(mask1_file)
    if not os.path.exists(mask2_file):
        os.makedirs(mask2_file)
    cv2.imwrite(warp1_file + end_name, warp_image1)
    cv2.imwrite(warp2_file + end_name, warp_image2)
    cv2.imwrite(mask1_file + end_name, mask1_warp)
    cv2.imwrite(mask2_file + end_name, mask2_warp)
    cv2.destroyAllWindows()

class UDIS_composition():
    def __init__(self,datapath,gpu,weight_path,LIIF,LIIF_models,ract_mask):
        self.datapath = datapath
        self.gpu = gpu
        self.weight_path = weight_path
        self.LIIF = LIIF
        self.LIIF_model = LIIF_models
        self.ract_mask = ract_mask


    def loadSingleData(self,datapath):
        # load image1
        warp1 = cv2.imread(datapath + "warp1.jpg")
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = cv2.imread(datapath + "warp2.jpg")
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = cv2.imread(datapath + "mask1.jpg")
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = mask1 / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = cv2.imread(datapath + "mask2.jpg")
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = mask2 / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        warp1_yuan = cv2.imread(datapath + "warp1_yuan.jpg")
        warp1_yuan = warp1_yuan.astype(dtype=np.float32)
        warp1_yuan = (warp1_yuan / 127.5) - 1.0
        warp1_yuan = np.transpose(warp1_yuan, [2, 0, 1])

        # load image2
        warp2_yuan = cv2.imread(datapath + "warp2_yuan.jpg")
        warp2_yuan = warp2_yuan.astype(dtype=np.float32)
        warp2_yuan = (warp2_yuan / 127.5) - 1.0
        warp2_yuan = np.transpose(warp2_yuan, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1).unsqueeze(0)
        warp2_tensor = torch.tensor(warp2).unsqueeze(0)
        mask1_tensor = torch.tensor(mask1).unsqueeze(0)
        mask2_tensor = torch.tensor(mask2).unsqueeze(0)

        return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, warp1_yuan, warp2_yuan

    def test_other(self,a,canvas):

        os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        # define the network
        net = Network()
        if torch.cuda.is_available():
            net = net.cuda()
        MODEL_DIR = self.weight_path
        # load the existing models if it exists
        ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
        ckpt_list.sort()
        if len(ckpt_list) != 0:
            model_path = ckpt_list[-1]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['model'])
            print('load model from {}!'.format(model_path))
        else:
            print('No checkpoint found!')
            return

        # load dataset(only one pair of images)
        warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, warp1_yuan, warp2_yuan = self.loadSingleData(self.datapath)
        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()

        net.eval()
        with torch.no_grad():
            batch_out = build_model(self, net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor,warp1_yuan,warp2_yuan,a,canvas)
        stitched_image = batch_out['stitched_image']
        learned_mask1 = batch_out['learned_mask1']
        learned_mask2 = batch_out['learned_mask2']
        enlarge_mask1 = batch_out['enlarged_mask1']
        enlarge_mask2 = batch_out['enlarged_mask2']

        path = self.datapath + "learn_mask1.jpg"
        cv2.imwrite(path, learned_mask1)
        path = self.datapath + "learn_mask2.jpg"
        cv2.imwrite(path, learned_mask2)
        path = self.datapath + "composition.jpg"
        cv2.imwrite(path, stitched_image)
        path = self.datapath + "enlarged_mask1.jpg"
        cv2.imwrite(path, enlarge_mask1)
        path = self.datapath + "enlarged_mask2.jpg"
        cv2.imwrite(path, enlarge_mask2)

        return stitched_image, enlarge_mask1
    def seam_loss_value(self, mask1, mask2, warp1, warp2):
        mask1 = mask1 / 255
        mask2 = mask2 / 255
        _, binary_mask1 = cv2.threshold(mask1, 0, 1, cv2.THRESH_BINARY)
        _, binary_mask2 = cv2.threshold(mask2, 0, 1, cv2.THRESH_BINARY)
        seam_mask = binary_mask1 * binary_mask2
        warp1 = np.transpose(warp1, [1, 2, 0])
        warp2 = np.transpose(warp2, [1, 2, 0])
        seam1 = (warp1 + 1.) * seam_mask - 1.
        seam2 = (warp2 + 1.) * seam_mask - 1.

        seam1_rgb = ((seam1 + 1.) * 127.5)
        seam2_rgb = ((seam2 + 1.) * 127.5)
        cv2.imwrite(self.datapath + "seam_loss1.jpg", seam1_rgb)
        cv2.imwrite(self.datapath + "seam_loss2.jpg", seam2_rgb)

        seam_loss = np.abs(seam1_rgb - seam2_rgb)
        cv2.imwrite(self.datapath + "seam_loss_.jpg", seam_loss)
        seam_loss = (np.sum(seam_loss))
        return seam_loss



def calculate_numpy_pixel_overlap_area(poly1, poly2):
    x_min = np.floor(np.min([np.min(poly1[:, 0]), np.min(poly2[:, 0])]))
    x_max = np.ceil(np.max([np.max(poly1[:, 0]), np.max(poly2[:, 0])]))
    y_min = np.floor(np.min([np.min(poly1[:, 1]), np.min(poly2[:, 1])]))
    y_max = np.ceil(np.max([np.max(poly1[:, 1]), np.max(poly2[:, 1])]))
def calculate_overlap_area(poly1, poly2):

    x_min = np.floor(np.min([np.min(poly1[:, 0]), np.min(poly2[:, 0])]))
    x_max = np.ceil(np.max([np.max(poly1[:, 0]), np.max(poly2[:, 0])]))
    y_min = np.floor(np.min([np.min(poly1[:, 1]), np.min(poly2[:, 1])]))
    y_max = np.ceil(np.max([np.max(poly1[:, 1]), np.max(poly2[:, 1])]))
    height, width = y_max, x_max
    canvas1 = np.zeros((int(height), int(width)), dtype=np.uint8)
    canvas2 = np.zeros((int(height), int(width)), dtype=np.uint8)

    cv2.fillConvexPoly(canvas1, poly1.astype(np.int32), 255)
    cv2.fillConvexPoly(canvas2, poly2.astype(np.int32), 255)

    intersection = cv2.bitwise_and(canvas1, canvas2)

    _, binary_intersection = cv2.threshold(intersection, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlap_area = 0
    for contour in contours:
        overlap_area += cv2.contourArea(contour)

    return overlap_area

def choose_keyframe(self, x, trans, td0, td1):
    x = x.reshape(-1, 2)
    max_ = -1
    j = 0
    H_translation = np.array([[1, 0, trans[0] - td0], [0, 1, trans[1] - td1], [0, 0, 1]])
    for i in range(len(self.keyframe_points)):
        if isinstance(self.keyframe_points[i], int) and self.keyframe_points[i] == 0:
            continue
        poly1 = x
        poly2 = cv2.perspectiveTransform(self.keyframe_points[i].reshape(-1, 1, 2), H_translation).reshape(-1,2)
        pixel_overlap_area = calculate_overlap_area(poly1, poly2)
        if max_ <  pixel_overlap_area:
            max_ = pixel_overlap_area
            j = i
        print("\nPixel Overlap Area:", pixel_overlap_area)
    print("\nmax_overlap:", max_)
    print("filename:", j)
    return max_, j

def choose_keyframe_many(self, x, trans, td0, td1):
    x = x.reshape(-1, 2)
    max_ = -1
    S = []
    j = 0
    H_translation = np.array([[1, 0, trans[0] - td0], [0, 1, trans[1] - td1], [0, 0, 1]])
    for i in range(len(self.keyframe_points)):
        if isinstance(self.keyframe_points[i], int) and self.keyframe_points[i] == 0:
            continue
        poly1 = x
        poly2 = cv2.perspectiveTransform(self.keyframe_points[i].reshape(-1, 1, 2), H_translation).reshape(-1,2)
        pixel_overlap_area = calculate_overlap_area(poly1, poly2)
        if max_ <  pixel_overlap_area:
            max_ = pixel_overlap_area
            j = i
        print("\nPixel Overlap Area:", pixel_overlap_area)
        base_img = cv2.imread(self.filename[i])
        base_pixel = np.count_nonzero(base_img[:,:,0])
        over_ratio = pixel_overlap_area / base_pixel
        if over_ratio > self.many_keyframe_H:
            S.append(i)
    print("\nmax_overlap:", max_)
    print("filename:", j)
    print("\nSelected indices:", S)
    return max_, j, S

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, viz2d
from lightglue.utils import load_image, rbd
def SIFT_GET_H(img1, img2):

    sift = cv2.SIFT_create(nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=20)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    dim, measure = 128, faiss.METRIC_L2
    # param = 'Flat'
    index = faiss.index_factory(dim, 'HNSW64', measure)
    des1 = np.ascontiguousarray(des1)
    des2 = np.ascontiguousarray(des2)
    index.add(des1)
    D, I = index.search(des2, 2)

    good = []
    for j in range(D.shape[0]):
        if D[j][0] < D[j][1] * 0.7:
            good.append(cv2.DMatch(j, I[j][0], 0, D[j][0]))
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

    return H, mask

def Glue_GET_point(img1, img2):

    ##LG and SuperPoint
    extractor = SuperPoint(max_num_keypoints=1000000).eval()  # load the extractor
    matcher = LightGlue(features='superpoint').eval() # load the matcher
    tensor1 = torch.from_numpy(img1)
    img1 = tensor1.permute(2, 0, 1)
    img1 = img1.to(torch.float32)
    img1 /= 255.0
    tensor2 = torch.from_numpy(img2)
    img2 = tensor2.permute(2, 0, 1)
    img2 = img2.to(torch.float32)
    img2 /= 255.0

    feats0 = extractor.extract(img1)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(img2)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0, m_kpts1 = m_kpts0.numpy(), m_kpts1.numpy()
    return m_kpts0, m_kpts1

def com_matchpoint(self, S, img):
    kp1_all = []
    kp2_all = []
    for i in S:
        img1_reset = cv2.imread(self.filename[i])
        H_reset = self.keyframe_H[i]
        kp1, kp2 = Glue_GET_point(img1_reset, img)

        points = np.array(kp1, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H_reset)
        kp1_trans = transformed_points.reshape(-1, 2)
        kp1_all.extend(kp1_trans)
        kp2_all.extend(kp2)
    kp1_all = np.array(kp1_all, dtype=np.float64)
    kp2_all = np.array(kp2_all, dtype=np.float64)
    return kp1_all, kp2_all



def upgrate_list_points(self, trans, td0, td1):
    H_translation = np.array([[1, 0, trans[0] - td0], [0, 1, trans[1] - td1], [0, 0, 1]])
    for i in range(len(self.keyframe_points)):
        if isinstance(self.keyframe_points[i], int) and self.keyframe_points[i] == 0:
            continue
        self.keyframe_points[i] = cv2.perspectiveTransform(self.keyframe_points[i].reshape(-1, 1, 2), H_translation)


def imageBlending(empty_matrix, output_img):
    # Convert images to uint8 for mask generation
    empty_matrix_uint8 = empty_matrix.astype(np.uint8)
    output_img_uint8 = output_img.astype(np.uint8)

    # Generate binary masks for non-zero pixels in both images
    mask1 = cv2.cvtColor(empty_matrix_uint8, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(output_img_uint8, cv2.COLOR_BGR2GRAY) > 0

    # Create a combined mask for the overlapping region
    overlap_mask = np.logical_and(mask1, mask2)

    # Create the result image initialized as zeros
    result_image = np.zeros_like(empty_matrix)

    # Process non-overlapping regions
    result_image[mask1 & ~overlap_mask] = empty_matrix[mask1 & ~overlap_mask]
    result_image[mask2 & ~overlap_mask] = output_img[mask2 & ~overlap_mask]

    # Process the overlapping region with 0.5 weight for both images
    for i in range(3):  # Iterate over the color channels (B, G, R)
        result_image[overlap_mask, i] = (empty_matrix[overlap_mask, i] * 0.5 + output_img[overlap_mask, i] * 0.5)

    # Convert the blended image back to uint8
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return result_image


def extract_edges(mask, border_width=1):

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    edges = cv2.Canny(mask, 100, 200)

    if border_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (border_width, border_width))
        edges = cv2.dilate(edges, kernel)
    return  cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def update_combined_edges(combined_edges, new_mask, border_width=1):

    new_edges = extract_edges(new_mask, border_width)

    combined_edges = cv2.bitwise_or(combined_edges, new_edges)

    return combined_edges


class HomographyCostFunction(pyceres.CostFunction):
    def __init__(self, src_points, dst_points):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(len(src_points) * 2)
        self.set_parameter_block_sizes([9])

        self.src_points = src_points
        self.dst_points = dst_points

    def Evaluate(self, parameters, residuals, jacobians):
        H = np.array(parameters[0]).reshape(3, 3)

        idx = 0
        for i, (src_point, dst_point) in enumerate(zip(self.src_points, self.dst_points)):
            src_homogeneous = np.array([src_point[0], src_point[1], 1.0])

            dst_predicted = H @ src_homogeneous
            dst_predicted /= dst_predicted[2]

            residuals[idx] = dst_predicted[0] - dst_point[0]
            residuals[idx + 1] = dst_predicted[1] - dst_point[1]
            idx += 2

            if jacobians is not None:
                J = np.zeros((2, 9))

                z2 = dst_predicted[2] ** 2
                J[0, 0] = -src_homogeneous[0] * dst_predicted[0] / z2
                J[0, 1] = -src_homogeneous[1] * dst_predicted[0] / z2
                J[0, 2] = -dst_predicted[0] / z2
                J[0, 3] = src_homogeneous[0] / dst_predicted[2]
                J[0, 4] = src_homogeneous[1] / dst_predicted[2]
                J[0, 5] = 1 / dst_predicted[2]

                J[1, 0] = -src_homogeneous[0] * dst_predicted[1] / z2
                J[1, 1] = -src_homogeneous[1] * dst_predicted[1] / z2
                J[1, 2] = -dst_predicted[1] / z2
                J[1, 3] = src_homogeneous[0] / dst_predicted[2]
                J[1, 4] = src_homogeneous[1] / dst_predicted[2]
                J[1, 5] = 1 / dst_predicted[2]

                jacobians[0][i * 2 * 9:(i * 2 + 1) * 9] = J[0]
                jacobians[0][(i * 2 + 1) * 9:(i * 2 + 2) * 9] = J[1]

        return True


def optimize_homography(src_points, dst_points, H_init):

    prob = pyceres.Problem()

    initial_homography = H_init.flatten()

    prob.add_parameter_block(initial_homography, 9)

    cost_function = HomographyCostFunction(src_points, dst_points)

    prob.add_residual_block(cost_function, None, [initial_homography])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.minimizer_type = pyceres.MinimizerType.TRUST_REGION
    options.minimizer_progress_to_stdout = False

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    H_optimized = initial_homography.reshape(3, 3)
    for i, src_point in enumerate(src_points):
        src_homogeneous = np.array([src_point[0], src_point[1], 1.0])
        dst_predicted = H_optimized @ src_homogeneous
        dst_predicted /= dst_predicted[2]

    return initial_homography.reshape(3, 3)


def dpm_mask(mask, point):
    color = 0
    thick = 4
    point = point.reshape(-1, 2)

    # 确保传入的是整数坐标
    cv2.line(mask, tuple(np.round(point[0]).astype(int)), tuple(np.round(point[1]).astype(int)), color, thick)
    cv2.line(mask, tuple(np.round(point[0]).astype(int)), tuple(np.round(point[3]).astype(int)), color, thick)
    cv2.line(mask, tuple(np.round(point[1]).astype(int)), tuple(np.round(point[2]).astype(int)), color, thick)
    cv2.line(mask, tuple(np.round(point[2]).astype(int)), tuple(np.round(point[3]).astype(int)), color, thick)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask