from typing import Optional
import torch
import os
from typing import List
import numpy as np
from PIL import Image
import cv2
import time
import sys

from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
)
from controlnet_aux import HEDdetector, OpenposeDetector, MLSDdetector


AUX_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "fusing/stable-diffusion-v1-5-controlnet-depth",
    "normal": "fusing/stable-diffusion-v1-5-controlnet-normal",
    "hed": "fusing/stable-diffusion-v1-5-controlnet-hed",
    "scribble": "fusing/stable-diffusion-v1-5-controlnet-scribble",
    "mlsd": "fusing/stable-diffusion-v1-5-controlnet-mlsd",
    "seg": "fusing/stable-diffusion-v1-5-controlnet-seg",
    "openpose": "fusing/stable-diffusion-v1-5-controlnet-openpose",
}

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"

if not os.path.exists(CONTROLNET_CACHE):
    print("controlnet cache missing, use `cog run script/download_weights` to download")

if not os.path.exists(SD15_WEIGHTS):
    print("sd15 weights missing, use `cog run python` and then load and save_pretrained('weights')")

class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        st = time.time()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD15_WEIGHTS,
            torch_dtype=torch.float16,
            local_files_only=True).to("cuda")

        self.controlnets = {}
        for name in AUX_IDS.keys():
            self.controlnets[name] = ControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, name),
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")

        # Depth + Normal
        self.depth_estimator = pipeline("depth-estimation")

        # Normal
        self.controlnet_hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

        # Hough
        self.mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

        # Seg
        self.controlnet_seg_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.controlnet_seg_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )

        # Pose
        self.controlnet_pose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        print("Setup complete in %f" % (time.time() - st))


    @torch.inference_mode()
    def predict(self,
                image: Path = Input(
                    description="Input image"
                ),
                prompt: str = Input(
                    description="Prompt for the model"
                ),
                structure: str = Input( # FIXME
                    description="Structure to condition on",
                    choices=["canny", "depth", "hed", "hough", "normal", "pose", "scribble", "seg"]
                ),
                num_samples: str = Input( # FIXME
                    description="Number of samples (higher values may OOM)",
                    choices=['1', '4'],
                    default='1'
                ),
                image_resolution: str = Input(
                    description="Resolution of image (square)",
                    choices=['256', '512', '768'],
                    default='512'
                ),
                steps: int = Input(
                    description="Steps",
                    default=20
                ),
                scale: float = Input(
                    description="Scale for classifier-free guidance",
                    default=9.0,
                    ge=0.1,
                    le=30.0
                ),
                seed: int = Input(
                    description="Seed",
                    default=None
                ),
                eta: float = Input(
                    description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
                    default=0.0
                ),
                a_prompt: str = Input(# FIXME
                    description="Additional text to be appended to prompt",
                    default="Best quality, extremely detailed"
                ),
                n_prompt: str = Input(# FIXME
                    description="Negative prompt",
                    default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                ),
                # Only applicable when model type is 'canny'
                low_threshold: int = Input(
                    description="[canny only] Line detection low threshold",
                    default=100,
                    ge=1,
                    le=255
                ),
                # Only applicable when model type is 'canny'
                high_threshold: int = Input(
                    description="[canny only] Line detection high threshold",
                    default=200,
                    ge=1,
                    le=255
                ),
                ) -> List[Path]:
        pipe = self.select_pipe(structure)
        pipe.enable_xformers_memory_efficient_attention()

        num_samples = int(num_samples)
        image_resolution = int(image_resolution)
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Load input_image
        input_image = Image.open(image)
        input_image = self.process_image(input_image, structure, low_threshold=low_threshold, high_threshold=high_threshold)

        # Why a_prompt?
        prompt = prompt + ', ' + a_prompt
        outputs = pipe(
            prompt,
            input_image,
            height=image_resolution,
            width=image_resolution,
            num_inference_steps=steps,
            guidance_scale=scale,
            eta=eta,
            negative_prompt=n_prompt,
            num_images_per_prompt=num_samples,
        )
        output_paths = []
        for i, sample in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths

    def select_pipe(self, structure):
        return StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=self.controlnets[structure],
            )

    def process_image(self, image, structure, low_threshold=100, high_threshold=200):
        if structure == 'canny':
            input_image = self.canny_preprocessor(image, low_threshold, high_threshold)
        elif structure == 'depth':
            input_image = self.depth_preprocessor(image)
        elif structure == 'hed':
            input_image = self.hed_preprocessor(image)
        elif structure == 'hough':
            input_image = self.hough_preprocessor(image)
        elif structure == 'normal':
            input_image = self.normal_preprocessor(image)
        elif structure == 'pose':
            input_image = self.pose_preprocessor(image)
        elif structure == 'scribble':
            input_image = self.scribble_preprocessor(image)
        elif structure == 'seg':
            input_image = self.seg_preprocessor(image)
        return input_image

    def canny_preprocessor(self, image, low_threshold, high_threshold):
        # Convert to numpy
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image

    def depth_preprocessor(self, image):
        image = self.depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def hed_preprocessor(self, image):
        # Convert to numpy
        image = np.array(image)
        return self.controlnet_hed(image)

    def hough_preprocessor(self, image):
        # Convert to numpy
        image = np.array(image)
        return self.mlsd(image)

    def normal_preprocessor(self, image):
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        bg_threhold = 0.4
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def pose_preprocessor(self, image):
        # Convert to numpy
        image = np.array(image)
        return self.controlnet_pose(image)

    def scribble_preprocessor(self, image):
        # Convert to numpy
        image = np.array(image)
        return self.controlnet_hed(image, scribble=True)

    def seg_preprocessor(self, image):
        image = image.convert('RGB')
        pixel_values = self.controlnet_seg_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.controlnet_seg_segmentor(pixel_values)
        seg = self.controlnet_seg_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

