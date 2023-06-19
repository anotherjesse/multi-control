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
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    CannyDetector,
)
from controlnet_aux.util import ade_palette
from midas_hack import MidasDetector


AUX_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "fusing/stable-diffusion-v1-5-controlnet-depth",
    "normal": "fusing/stable-diffusion-v1-5-controlnet-normal",
    "hed": "fusing/stable-diffusion-v1-5-controlnet-hed",
    "scribble": "fusing/stable-diffusion-v1-5-controlnet-scribble",
    "hough": "fusing/stable-diffusion-v1-5-controlnet-mlsd",
    "seg": "fusing/stable-diffusion-v1-5-controlnet-seg",
    "pose": "fusing/stable-diffusion-v1-5-controlnet-openpose",
    "qr": "DionTimmer/controlnet_qrcode-control_v1p_sd15",
}

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}


SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"
PROCESSORS_CACHE = "processors-cache"
MISSING_WEIGHTS = []

if not os.path.exists(CONTROLNET_CACHE) or not os.path.exists(PROCESSORS_CACHE):
    print(
        "controlnet weights missing, use `cog run python script/download_weights` to download"
    )
    MISSING_WEIGHTS.append("controlnet")

if not os.path.exists(SD15_WEIGHTS):
    print(
        "sd15 weights missing, use `cog run python` and then load and save_pretrained('weights')"
    )
    MISSING_WEIGHTS.append("sd15")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if len(MISSING_WEIGHTS) > 0:
            print("skipping setup... missing weights: ", MISSING_WEIGHTS)
            return

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

        self.canny = CannyDetector()

        # Depth + Normal
        self.midas = MidasDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.hed = HEDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        # Hough
        self.mlsd = MLSDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.seg_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )
        self.seg_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small", cache_dir=PROCESSORS_CACHE
        )

        self.pose = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=PROCESSORS_CACHE
        )

        print("Setup complete in %f" % (time.time() - st))

    def canny_preprocess(self, img):
        return self.canny(img)

    def depth_preprocess(self, img):
        return self.midas(img)

    def hough_preprocess(self, img):
        return self.mlsd(img)

    def normal_preprocess(self, img):
        return self.midas(img, depth_and_normal=True)[1]

    def scribble_preprocess(self, img):
        return self.hed(img, scribble=True)

    def qr_preprocess(self, img):
        return img

    def pose_preprocess(self, img):
        return self.pose(img)

    def hed_preprocess(self, img):
        return self.hed(img)

    def seg_preprocess(self, image):
        image = image.convert("RGB")
        pixel_values = self.seg_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.seg_segmentor(pixel_values)
        seg = self.seg_processor.post_process_semantic_segmentation(
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

    def build_pipe(
        self, inputs, low_threshold=100, high_threshold=200, guess_mode=False
    ):
        control_nets = []
        processed_control_images = []
        conditioning_scales = []

        for input in inputs.items():
            if input["image"] is None or input["name"] is None:
                continue
            name = input["name"]
            control_nets.append(self.controlnets[name])
            image = input["image"]
            img = Image.open(image)
            if name == "canny":
                img = self.canny_preprocess(img, low_threshold, high_threshold)
            else:
                img = getattr(self, "{}_preprocess".format(name))(img)

            processed_control_images.append(img)
            if input["conditioning_scale"] is None:
                conditioning_scale = 1.0
            else:
                conditioning_scale = input["conditioning_scale"]
            conditioning_scales.append(conditioning_scale)

        if len(control_nets) == 0:
            pipe = self.pipe
            kwargs = {}
        else:
            pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=control_nets,
            )
            kwargs = {
                "image": processed_control_images,
                "controlnet_conditioning_scale": conditioning_scales,
                "guess_mode": guess_mode,
            }

        return pipe, kwargs

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for the model"),
        controlnet_1: str = Input(
            description="Structure of controlnet", default=None, choices=AUX_IDS.keys()
        ),
        controlnet_1_image: Path = Input(
            description="Control image for controlnet", default=None
        ),
        controlnet_1_conditioning_scale: float = Input(
            description="override scale for controlnet", default=None
        ),
        controlnet_2: str = Input(
            description="Structure of controlnet", default=None, choices=AUX_IDS.keys()
        ),
        controlnet_2_image: Path = Input(
            description="Control image for controlnet", default=None
        ),
        controlnet_2_conditioning_scale: float = Input(
            description="override scale for controlnet", default=None
        ),
        controlnet_3: str = Input(
            description="Structure of controlnet", default=None, choices=AUX_IDS.keys()
        ),
        controlnet_3_image: Path = Input(
            description="Control image for controlnet", default=None
        ),
        controlnet_3_conditioning_scale: float = Input(
            description="override scale for controlnet", default=None
        ),
        controlnet_4: str = Input(
            description="Structure of controlnet", default=None, choices=AUX_IDS.keys()
        ),
        controlnet_4_image: Path = Input(
            description="Control image for controlnet", default=None
        ),
        controlnet_4_conditioning_scale: float = Input(
            description="override scale for controlnet", default=None
        ),
        num_outputs: int = Input(
            description="Number of images to generate",
            ge=1,
            le=10,
            default=1,
        ),
        image_resolution: int = Input(
            description="Resolution of image (smallest dimension)",
            choices=[256, 512, 768],
            default=512,
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        num_inference_steps: int = Input(
            description="Steps to run denoising", default=20
        ),
        guidance_scale: float = Input(
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
        negative_prompt: str = Input(  # FIXME
            description="Negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        # Only applicable when using 'canny'
        low_threshold: int = Input(
            description="[canny only] Line detection low threshold`",
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
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
    ) -> List[Path]:
        if len(MISSING_WEIGHTS) > 0:
            raise Exception("missing weights")

        pipe, kwargs = self.build_pipe(
            [
                {
                    "name": controlnet_1,
                    "image": controlnet_1_image,
                    "conditioning_scale": controlnet_1_conditioning_scale,
                },
                {
                    "name": controlnet_2,
                    "image": controlnet_2_image,
                    "conditioning_scale": controlnet_2_conditioning_scale,
                },
                {
                    "name": controlnet_3,
                    "image": controlnet_3_image,
                    "conditioning_scale": controlnet_3_conditioning_scale,
                },
                {
                    "name": controlnet_4,
                    "image": controlnet_4_image,
                    "conditioning_scale": controlnet_4_conditioning_scale,
                },
            ],
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            guess_mode=guess_mode,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if "image" in kwargs:
            img = kwargs["image"][0]
            scale = float(image_resolution) / (min(img.size))

            def quick_rescale(dim):
                """quick rescale to a multiple of 64, as per original controlnet"""
                dim *= scale
                return int(np.round(dim / 64.0)) * 64

            width = quick_rescale(img.size[0])
            height = quick_rescale(img.size[1])
        else:
            width = height = image_resolution

        generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            pipe.safety_checker = None

        result_count = 0
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)

            output = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                **kwargs,
            )

            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue

            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            yield Path(output_path)
            result_count += 1

        if result_count == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
