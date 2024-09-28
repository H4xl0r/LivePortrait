# coding: utf-8
"""
for human
"""
import tyro
import os
import os.path as osp
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal
from src.live_portrait_pipeline_webcam import LivePortraitPipelineWebcam
from src.live_portrait_pipeline_webcam_animal import LivePortraitPipelineWebcamAnimal



import cv2
import numpy as np
import time

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

# Function to get available webcam devices
def get_webcam_devices():
    return [i for i in range(10) if cv2.VideoCapture(i).isOpened()]


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

    # Check the run_mode
    if args.run_mode == 'animal':
        #Animal

        #Webcam
        if args.input_type == 'webcam':

            # Handle webcam
            live_portrait_pipeline_webcam_animal = LivePortraitPipelineWebcamAnimal(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
            )

            #Get Devices
            devices = get_webcam_devices()

            # Initialize webcam
            if not devices:
                print("No webcam devices found.")
                return
            else:
                print("Available webcam devices:", devices)
                selected_device = int(input("Select a webcam ID from the list above: "))

            # Check if the selected device is valid
            if selected_device in devices:
                cap = cv2.VideoCapture(selected_device)
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                    return
                try:
                    # Process the first frame to initialize
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        return

                    source_image_path = args.source  # Set the source image path here
                    x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb = live_portrait_pipeline_webcam_animal.execute_frame(frame, source_image_path)

                    while True:
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame.")
                            break

                        # Process the frame
                        result = live_portrait_pipeline_webcam_animal.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, frame)

                        # Display images
                        cv2.imshow('img_rgb Image', img_rgb)
                        cv2.imshow('Source Frame', frame)

                        # Convert the result from RGB to BGR before displaying
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Live Portrait', result_bgr)

                        # Press 'q' to exit the loop
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
            else:
                print("Invalid device selected.")
        else:
            live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg
            )
            live_portrait_pipeline_animal.execute(args)
    else:
        #Human

        #Webcam
        if args.input_type == 'webcam':

            # Handle webcam
            live_portrait_pipeline_webcam = LivePortraitPipelineWebcam(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
            )

            #Get Devices
            devices = get_webcam_devices()

            # Initialize webcam
            if not devices:
                print("No webcam devices found.")
                return
            else:
                print("Available webcam devices:", devices)
                selected_device = int(input("Select a webcam ID from the list above: "))

            # Check if the selected device is valid
            if selected_device in devices:
                cap = cv2.VideoCapture(selected_device)
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                    return
                try:
                    # Process the first frame to initialize
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        return

                    source_image_path = args.source  # Set the source image path here
                    x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb = live_portrait_pipeline_webcam.execute_frame(frame, source_image_path)

                    while True:
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame.")
                            break

                        # Process the frame
                        result = live_portrait_pipeline_webcam.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, frame)

                        # Display images
                        cv2.imshow('img_rgb Image', img_rgb)
                        cv2.imshow('Source Frame', frame)

                        # Convert the result from RGB to BGR before displaying
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Live Portrait', result_bgr)

                        # Press 'q' to exit the loop
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
            else:
                print("Invalid device selected.")

        else:
            # Handle default mode
            live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
            )
            live_portrait_pipeline.execute(args)


if __name__ == "__main__":
    st = time.time()
    main()
    print("Generation time:", (time.time() - st) * 1000)
