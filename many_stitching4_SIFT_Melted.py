import argparse
import os
import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
import functional_ as fun

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_grad_enabled(False)

class stitching(object):
    def __init__(self, args, _GMS, COMP, cuda, weight_path_SP, weight_path_COMP):
        self.log_path = args.log_path
        self.folder_path = args.file_path
        self.filename = []
        self.COMP=COMP
        self.cuda = cuda
        self.weight_path_SP = weight_path_SP
        self.weight_path_COMP = weight_path_COMP
        self.args = args
        self.LIIF = args.LIIF
        self.LIIF_model = args.LIIF_model

        image_extensions = ['.jpg', '.jpeg', '.png']
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path) and any(ext in filename.lower() for ext in image_extensions):
                self.filename.append(file_path)
        self.len = len(self.filename)
        self._GMS = _GMS
        self.scale_factor = 0.7
        self.over_ratio = 1
        self.key_frame = False
        self.ract_mask = False
        self.light_ave = False
        self.keyframe_points = self.len * [0]
        self.keyframe_H = self.len * [0]

    def SuperPoint(self):
        ract_mask = self.ract_mask
        key_frame = self.key_frame
        light_ave = self.light_ave
        log_path = self.log_path
        H_result = np.eye(3)

        img1_ = cv2.imread(self.filename[0])
        sizer1 = [img1_.shape[0], img1_.shape[1]]
        rows, cols = img1_.shape[:2]
        base_point = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]]).reshape(-1, 1, 2)  # 原始四点坐标
        self.keyframe_points[0] = base_point.reshape(-1,2)
        self.keyframe_H[0] = np.eye(3)
        # SP = fun.SuperPointFrontend(
        #     weights_path=self.weight_path_SP,
        #     nms_dist=4,
        #     conf_thresh=0.015,
        #     nn_thresh=0.7,
        #     cuda=self.cuda,
        #     device=self.args.gpu)

        td1 = 0
        td0 = 0
        [x_min, y_min] = [1000, 1000]
        [x_max, y_max] = [0, 0]
        out_p = img1_
        out_full = img1_
        out_full_mask = np.ones_like(img1_) * 255
        next_base_mask = np.ones_like(img1_) * 255
        out_p_COMP = img1_
        for i in range(0, self.len-1):
            s0_time = datetime.datetime.now()
            img2_ = cv2.imread(self.filename[i + 1])
            sizer2 = [img2_.shape[0], img2_.shape[1]]
            rows, cols = img2_.shape[:2]
            target_point = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]]).reshape(-1, 1, 2)  # 原始四点坐标
            H, mask = fun.SIFT_GET_H(img1_, img2_)

            H_copy = H_result.copy()
            H_result = np.dot(H, H_result)
            target_warp_point = cv2.perspectiveTransform(target_point, H_result)
            x_min_copy, x_max_copy, y_min_copy, y_max_copy = x_min, x_max, y_min, y_max
            x_min = np.int32(min(x_min, np.min(base_point[:, :, 0]), np.min(target_warp_point[:, :, 0]))-0.5)
            x_max = np.int32(max(x_max, np.max(base_point[:, :, 0]), np.max(target_warp_point[:, :, 0]))+0.5)
            y_min = np.int32(min(y_min, np.min(base_point[:, :, 1]), np.min(target_warp_point[:, :, 1]))-0.5)
            y_max = np.int32(max(y_max, np.max(base_point[:, :, 1]), np.max(target_warp_point[:, :, 1]))+0.5)
            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
            output_img = cv2.warpPerspective(img2_, H_translation.dot(H_result),
                                             (x_max - x_min, y_max - y_min))
            target_warp_point = cv2.perspectiveTransform(target_point, H_translation.dot(H_result))
            choose_max_overlap, filename_path = fun.choose_keyframe(self, target_warp_point_clipped, translation_dist, td0, td1)
            print("key_frame:", key_frame)
            if key_frame:
                if filename_path != i:
                    img1_reset = cv2.imread(self.filename[filename_path])
                    rows, cols = img1_reset.shape[:2]
                    target_point = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]]).reshape(-1, 1, 2)  # 原始四点坐标
                    H, mask = fun.SIFT_GET_H(img1_reset, img2_)
                    H_result = np.dot(H, self.keyframe_H[filename_path])
                    target_warp_point = cv2.perspectiveTransform(target_point, H_result)
                    x_min = np.int32(min(x_min_copy, np.min(base_point[:, :, 0]), np.min(target_warp_point[:, :, 0])) - 0.5)
                    x_max = np.int32(max(x_max_copy, np.max(base_point[:, :, 0]), np.max(target_warp_point[:, :, 0])) + 0.5)
                    y_min = np.int32(min(y_min_copy, np.min(base_point[:, :, 1]), np.min(target_warp_point[:, :, 1])) - 0.5)
                    y_max = np.int32(max(y_max_copy, np.max(base_point[:, :, 1]), np.max(target_warp_point[:, :, 1])) + 0.5)
                    translation_dist = [-x_min, -y_min]
                    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
                    output_img = cv2.warpPerspective(img2_, H_translation.dot(H_result),
                                                     (x_max - x_min, y_max - y_min))
                    target_warp_point = cv2.perspectiveTransform(target_point, H_translation.dot(H_result))
                    target_warp_point_clipped = np.clip(target_warp_point, 0, None)


            mask = np.ones_like(img2_, dtype=np.uint8) * 255
            mask = mask[:, :, 0]
            transformed_mask = cv2.warpPerspective(mask, H_translation.dot(H_result),
                                                   (x_max - x_min, y_max - y_min))
            inverted_mask = cv2.bitwise_not(transformed_mask)

            empty_matrix = np.zeros_like(output_img)
            empty_matrix_COMP = np.zeros_like(output_img)
            empty_matrix_COMP_mask = np.zeros_like(output_img)
            empty_matrix_COMP_base = np.zeros_like(output_img)

            empty_matrix[translation_dist[1] - td1: translation_dist[1] - td1 + out_p.shape[0],
            translation_dist[0] - td0: translation_dist[0] - td0 + out_p.shape[1]] = out_p

            empty_matrix_COMP_mask[translation_dist[1] - td1: translation_dist[1] - td1 + out_p.shape[0],
            translation_dist[0] - td0: translation_dist[0] - td0 + out_p.shape[1]] = next_base_mask

            base_pixel = np.count_nonzero(transformed_mask)
            over_ratio = choose_max_overlap / base_pixel
            print("over_ratio:{}".format(over_ratio))

            if key_frame:
                fun.upgrate_list_points(self, translation_dist, td0, td1)

                if over_ratio < self.over_ratio:
                    self.keyframe_points[i + 1] = target_warp_point_clipped.reshape(-1,2)
                    self.keyframe_H[i + 1] = H_result.reshape(-1, 3)
                    print("keyframe:", self.filename[filename_path])

            if self.COMP:
                empty_matrix_COMP[translation_dist[1] - td1: translation_dist[1] - td1 + out_p_COMP.shape[0],
                translation_dist[0] - td0: translation_dist[0] - td0 + out_p_COMP.shape[1]] = out_p_COMP

                empty_matrix_COMP_base[translation_dist[1] - td1: translation_dist[1] - td1 + out_p_COMP.shape[0],
                translation_dist[0] - td0: translation_dist[0] - td0 + out_p_COMP.shape[1]] = out_full

                base_rows, base_cols = empty_matrix_COMP_base.shape[:2]
                target_warp_point_clipped_reshape = target_warp_point_clipped.reshape(-1,2)
                rect_min_x, rect_max_x, rect_min_y, rect_max_y= (np.int32(min(target_warp_point_clipped_reshape[:, 0])-0.5),
                                                                 np.int32(max(target_warp_point_clipped_reshape[:, 0])+0.5),
                                                                 np.int32(min(target_warp_point_clipped_reshape[:, 1])-0.5),
                                                                 np.int32(max(target_warp_point_clipped_reshape[:, 1])+0.5))
                expand_rect_points = np.float32([[rect_min_x, rect_min_y], [rect_min_x, rect_max_y],
                                                 [rect_max_x, rect_max_y], [rect_max_x, rect_min_y]]).reshape(-1, 2)
                canvas = np.zeros((int(base_rows), int(base_cols)), dtype=np.uint8)
                _, out_full_mask = cv2.threshold(empty_matrix_COMP_base, 1, 255, cv2.THRESH_BINARY)
                out_full_mask = cv2.cvtColor(out_full_mask, cv2.COLOR_BGR2GRAY)
                out_full_mask = cv2.GaussianBlur(out_full_mask, (5, 5), 0)
                cv2.imwrite(log_path + 'out_full_mask.jpg', out_full_mask)

                cv2.fillConvexPoly(canvas, expand_rect_points.astype(np.int32), 255)
                cv2.imwrite(log_path + 'a.jpg', empty_matrix_COMP)
                cv2.imwrite(log_path + 'aa.jpg', empty_matrix_COMP_base)
                cv2.imwrite(log_path + 'expand_rect_mask.jpg', canvas)



            out = fun.imageBlending(empty_matrix, output_img)

            transformed_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_GRAY2BGR)
            td1 = translation_dist[1]
            td0 = translation_dist[0]
            out_p = out
            cv2.imwrite(log_path + 'ponoa.jpg', out_p)
            s1_time = datetime.datetime.now()
            if self.COMP:

                expand_mask1 = cv2.bitwise_and(canvas, out_full_mask)

                expand_mask1 = cv2.cvtColor(expand_mask1, cv2.COLOR_GRAY2BGR)

                if light_ave:
                    image1_clip = cv2.bitwise_and(empty_matrix_COMP_base, expand_mask1)
                    image2_clip = cv2.bitwise_and(output_img, expand_mask1)
                    cv2.imwrite(log_path + "light_clip1.jpg", image1_clip)
                    cv2.imwrite(log_path + "light_clip2.jpg", image2_clip)

                    image1_clip = image1_clip.astype(float)
                    image2_clip = image2_clip.astype(float)
                    image1_float = empty_matrix_COMP_base.astype(float)
                    image2_float = output_img.astype(float)

                    black_mask1 = np.all(image1_clip < 1, axis=2)
                    black_mask2 = np.all(image2_clip < 1, axis=2)
                    image1_clip[black_mask1] = np.nan
                    image2_clip[black_mask2] = np.nan
                    brightness1 = np.nanmean(image1_clip)
                    brightness2 = np.nanmean(image2_clip)
                    average_brightness = (brightness1 + brightness2) / 2
                    image1_float *= average_brightness / brightness1
                    image2_float *= average_brightness / brightness2
                    empty_matrix_COMP_base = cv2.convertScaleAbs(image1_float)
                    output_img = cv2.convertScaleAbs(image2_float)

                base_img_ = (empty_matrix_COMP_base / 127.5) - 1.0
                mask_x = (expand_mask1 / 255)
                clip_img = (base_img_ + 1.) * mask_x - 1.
                expand_warp1 = ((clip_img + 1.) * 127.5)
                print("ract_mask:",ract_mask)
                if ract_mask:
                    cv2.imwrite(log_path + 'mask1_yuan.jpg', expand_mask1)
                    cv2.imwrite(log_path + 'warp1_yuan.jpg', expand_warp1)
                    # cv2.imwrite(log_path + 'mask1_yuan.jpg', empty_matrix_COMP_mask)
                    # cv2.imwrite(log_path + 'warp1_yuan.jpg', empty_matrix_COMP)
                    cv2.imwrite(log_path + 'mask2_yuan.jpg', transformed_mask)
                    cv2.imwrite(log_path + 'warp2_yuan.jpg', output_img)
                    cv2.imwrite(log_path + "all_mask.jpg", cv2.bitwise_or(transformed_mask, expand_mask1))
                else:
                    cv2.imwrite(log_path + 'mask1_yuan.jpg', empty_matrix_COMP_mask)
                    cv2.imwrite(log_path + 'warp1_yuan.jpg', empty_matrix_COMP)
                    cv2.imwrite(log_path + 'mask2_yuan.jpg', transformed_mask)
                    cv2.imwrite(log_path + 'warp2_yuan.jpg', output_img)
                    cv2.imwrite(log_path + "all_mask.jpg", cv2.bitwise_or(transformed_mask, empty_matrix_COMP_mask))

                over_mask2 = cv2.bitwise_and(expand_mask1, transformed_mask)
                over_img = cv2.bitwise_and(expand_warp1.astype(np.uint8), over_mask2)
                cv2.imwrite(log_path + 'over_img.jpg', over_img)


                image_filenames = ['mask1', 'warp1', 'mask2', 'warp2']
                for filename in image_filenames:
                    original_image = Image.open(log_path + f'{filename}_yuan.jpg')
                    new_width = int(original_image.width * self.scale_factor)
                    new_height = int(original_image.height * self.scale_factor)
                    smaller_image = original_image.resize((new_width, new_height), Image.LANCZOS)
                    # if filename.startswith('mask'):
                    #     smaller_image = smaller_image.filter(ImageFilter.EDGE_ENHANCE)
                    smaller_image.save(log_path + f'{filename}.jpg')
                    original_image.close()

                s2_time = datetime.datetime.now()
                UDIS_COMP = fun.UDIS_composition(self.log_path, self.args.gpu, weight_path=self.weight_path_COMP, LIIF=self.LIIF, LIIF_models=self.LIIF_model, ract_mask= ract_mask)
                COMP_img,xxxx,= UDIS_COMP.test_other(self.scale_factor, canvas)


                if ract_mask:
                    clip_base_mask = cv2.bitwise_not(cv2.bitwise_or(expand_mask1, transformed_mask))
                else:
                    clip_base_mask = cv2.bitwise_not(cv2.bitwise_or(empty_matrix_COMP_mask, transformed_mask))

                base_img_ = (empty_matrix_COMP_base / 127.5) - 1.0
                mask_x = (clip_base_mask / 255)
                clip_img = (base_img_ + 1.) * mask_x - 1.
                clip_base = ((clip_img + 1.) * 127.5)

                cv2.imwrite(log_path + 'clip_base_mask.jpg', clip_base_mask)
                cv2.imwrite(log_path + 'COMP_img.jpg', COMP_img.astype(np.uint8))
                cv2.imwrite(log_path + 'clip_base.jpg', clip_base)

                out_p_COMP = cv2.add(COMP_img.astype(np.uint8), clip_base.astype(np.uint8))
                cv2.imwrite(log_path + 'result.jpg', out_p_COMP)
                cv2.imwrite(log_path + 'result' + '{}'.format(i) + '.jpg', out_p_COMP)
                s3_time = datetime.datetime.now()


                xxxx = cv2.add(xxxx.astype(np.uint8), clip_base_mask.astype(np.uint8))
                cv2.imwrite(log_path + 'xxxxx.jpg', xxxx)

                out_full = out_p_COMP.copy()

                base_img_ = (out_p_COMP / 127.5) - 1.0
                mask_x = (transformed_mask / 255)
                clip_img = (base_img_ + 1.) * mask_x - 1.
                out_p_COMP = ((clip_img + 1.) * 127.5)

                cv2.imwrite(log_path + 'out_p_COMP.jpg', out_p_COMP)

                s0 = s0_time.strftime("%Y-%m-%d %H:%M:%S")
                s1 = s1_time.strftime("%Y-%m-%d %H:%M:%S")
                s2 = s2_time.strftime("%Y-%m-%d %H:%M:%S")
                s3 = s3_time.strftime("%Y-%m-%d %H:%M:%S")
                print("拼接时间：", s0, " ", s1)
                print("拼接缝修复时间：", s2, " ", s3)
            next_base_mask = transformed_mask
            img1_ = img2_
            sizer1 = [img1_.shape[0], img1_.shape[1]]
        tmp_H, tmp_W, _ = np.shape(out_p)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    # file_path = 'longlist'
    mask_width = 200
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='')   ##output
    parser.add_argument('--file_path', type=str, default='')  ##input
    parser.add_argument('--weight_path_SP', type=str, default='./stitching_models/weights/superpoint_v1.pth')
    parser.add_argument('--weight_path_COMP', type=str, default='./model')
    parser.add_argument('--COMP', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius'' (Must be positive)')
    parser.add_argument('--max_keypoints', type=int, default=-1,
            help='Maximum number of keypoints detected by Superpoint'' (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,
            help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--superglue', type=str, default='./stitching_models/weights/superglue_indoor.pth',
            help='SuperGlue weights')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
            help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2,
            help='SuperGlue match threshold')
    parser.add_argument('--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument('--LIIF', type=bool, default=False)
    parser.add_argument('--LIIF_model', type=str, default='./stitching_models/weights/edsr-baseline-liif.pth')
    parser.add_argument('--ract_mask', type=bool, default=False)
    args = parser.parse_args()

    stitching = stitching(args,_GMS=False,COMP=args.COMP,cuda=args.cuda,weight_path_SP=args.weight_path_SP, weight_path_COMP=args.weight_path_COMP)
    stitching.SuperPoint()

    final_time = datetime.datetime.now()




    format_begin_time = begin_time.strftime("%Y-%m-%d %H:%M:%S")
    format_final_time = final_time.strftime("%Y-%m-%d %H:%M:%S")
    # 打印格式化后的时间
    print("拼接开始时间：", format_begin_time)
    print("拼接结束时间：", format_final_time)