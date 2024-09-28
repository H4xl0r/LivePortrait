"""
Pipeline of LivePortrait
"""

import cv2
import numpy as np
import pickle
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit
from .utils.helper import mkdir, basename, dct2device, is_video, is_template
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapperAnimal



def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipelineWebcamAnimal:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper_animal = LivePortraitWrapperAnimal(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg, image_type='animal_face', flag_use_half_precision=inference_cfg.flag_use_half_precision)

    def execute_frame(self, frame, source_image_path):
        inf_cfg = self.live_portrait_wrapper_animal.inference_cfg

        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

        # Crop the image to focus on the animal face
        crop_info = self.cropper.crop_single_image(img_rgb)
        img_crop_256x256 = crop_info['img_crop_256x256']

        I_s = self.live_portrait_wrapper_animal.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper_animal.get_kp_info(I_s)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper_animal.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper_animal.transform_keypoint(x_s_info)

        return x_s, f_s, R_s, x_s_info, crop_info, img_rgb

    def generate_frame(self, x_s, f_s, R_s, x_s_info, crop_info, img_rgb, driving_frame):
        inference_cfg = self.live_portrait_wrapper_animal.inference_cfg

        # Process driving frame
        driving_rgb = cv2.resize(driving_frame, (256, 256))
        I_d_i = self.live_portrait_wrapper_animal.prepare_videos([driving_rgb])[0]

        x_d_i_info = self.live_portrait_wrapper_animal.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        # Calculate new transformation parameters
        delta_new = x_d_i_info['exp']
        t_new = x_d_i_info['t']
        t_new[..., 2].fill_(0)  # Zero tz
        scale_new = x_s_info['scale']

        x_d_i = scale_new * (x_s @ R_d_i + delta_new) + t_new

        out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
        I_p_i = self.live_portrait_wrapper_animal.parse_output(out['out'])[0]

        return I_p_i
