# coding: utf-8
"""
for human
"""

import os
import os.path as osp
import tyro
import subprocess
import cv2
import time
import numpy as np
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline_webcam import LivePortraitPipelineWebcam



def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

# Function to get available webcam devices
def get_webcam_devices():
    return [i for i in range(10) if cv2.VideoCapture(i).isOpened()]


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline_webcam = LivePortraitPipelineWebcam(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    # live_portrait_pipeline.execute(args)
    # Get available devices

    devices = get_webcam_devices()

    # Initialize webcam 'assets/examples/driving/d6.mp4'
    if not devices:
        print("No webcam devices found.")
        return
    else:
        print("Available webcam devices:", devices)
        selected_device = int(input("Select a webcam ID from the list above: "))

    # Check if the selected device is valid
    if selected_device in devices:
        cap = cv2.VideoCapture(selected_device)

    # Process the first frame to initialize
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return

    source_image_path = args.source_image  # Set the source image path here
    x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb = live_portrait_pipeline_webcam.execute_frame(frame, source_image_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame

        result = live_portrait_pipeline_webcam.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, frame)
        cv2.imshow('img_rgb Image', img_rgb)
        cv2.imshow('Source Frame', frame)


        # [Key Change] Convert the result from RGB to BGR before displaying
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


        # Display the resulting frame
        cv2.imshow('Live Portrait', result_bgr)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    st = time.time()
    main()
    print("Generation time:", (time.time() - st) * 1000)
