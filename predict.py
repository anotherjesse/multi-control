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
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    MidasDetector,
    CannyDetector,
)
from controlnet_aux.util import ade_palette


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
PROCESSORS_CACHE = "processors-cache"

if not os.path.exists(CONTROLNET_CACHE):
    print("controlnet cache missing, use `cog run script/download_weights` to download")

if not os.path.exists(SD15_WEIGHTS):
    print(
        "sd15 weights missing, use `cog run python` and then load and save_pretrained('weights')"
    )


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        st = time.time()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD15_WEIGHTS, torch_dtype=torch.float16, local_files_only=True
        ).to("cuda")

        self.controlnets = {}
        for name in AUX_IDS.keys():
            self.controlnets[name] = ControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, name),
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")

        # Canny
        self.canny = CannyDetector()

        # Depth + Normal
        self.midas = MidasDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        # HED
        self.hed = HEDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        # Hough
        self.mlsd = MLSDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        # Seg
        self.controlnet_seg_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )
        self.controlnet_seg_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )

        # Pose
        self.open_pose = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=PROCESSORS_CACHE
        )

        print("Setup complete in %f" % (time.time() - st))

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt for the model"),
        structure: str = Input(
            description="Structure to condition on",
            choices=[
                "canny",
                "depth",
                "hed",
                "hough",
                "normal",
                "pose",
                "scribble",
                "seg",
            ],
        ),
        num_samples: str = Input(  # FIXME
            description="Number of samples (higher values may OOM)",
            choices=["1", "4"],
            default="1",
        ),
        image_resolution: str = Input(
            description="Resolution of image (square)",
            choices=["256", "512", "768"],
            default="512",
        ),
        steps: int = Input(description="Steps", default=20),
        scale: float = Input(
            description="Scale for classifier-free guidance",
            default=9.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(
            description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
            default=0.0,
        ),
        a_prompt: str = Input(  # FIXME
            description="Additional text to be appended to prompt",
            default="Best quality, extremely detailed",
        ),
        n_prompt: str = Input(  # FIXME
            description="Negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        # Only applicable when model type is 'canny'
        low_threshold: int = Input(
            description="[canny only] Line detection low threshold",
            default=100,
            ge=1,
            le=255,
        ),
        # Only applicable when model type is 'canny'
        high_threshold: int = Input(
            description="[canny only] Line detection high threshold",
            default=200,
            ge=1,
            le=255,
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
        input_image = self.process_image(
            input_image,
            structure,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

        # Why a_prompt?
        prompt = prompt + ", " + a_prompt
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
        if structure == "canny":
            input_image = self.canny(image, low_threshold, high_threshold)
        elif structure == "depth":
            input_image = self.midas(image)
        elif structure == "hed":
            input_image = self.hed(image)
        elif structure == "hough":
            input_image = self.mlsd(image)
        elif structure == "normal":
            input_image = self.midas(image, depth_and_normal=True)[1]
        elif structure == "pose":
            input_image = self.open_pose(image)
        elif structure == "scribble":
            input_image = self.hed(image, scribble=True)
        elif structure == "seg":
            input_image = self.seg_preprocessor(image)
        return input_image

    def seg_preprocessor(self, image):
        image = image.convert("RGB")
        pixel_values = self.controlnet_seg_processor(
            image, return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            outputs = self.controlnet_seg_segmentor(pixel_values)
        seg = self.controlnet_seg_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        color_seg = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
        )  # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image
