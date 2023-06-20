from typing import Optional
import torch
import os
from typing import List
import numpy as np
from PIL import Image
import cv2
import time
import shutil
import sys

from torchvision.io.video import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import trange


from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
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

raft_transform = Raft_Large_Weights.DEFAULT.transforms()

AUX_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "fusing/stable-diffusion-v1-5-controlnet-depth",
    "normal": "fusing/stable-diffusion-v1-5-controlnet-normal",
    "hed": "fusing/stable-diffusion-v1-5-controlnet-hed",
    "scribble": "fusing/stable-diffusion-v1-5-controlnet-scribble",
    "hough": "fusing/stable-diffusion-v1-5-controlnet-mlsd",
    "pose": "fusing/stable-diffusion-v1-5-controlnet-openpose",
    "qr": "DionTimmer/controlnet_qrcode-control_v1p_sd15",
    "seg": "fusing/stable-diffusion-v1-5-controlnet-seg",
    "temporalnet2": "wav/TemporalNet2",
}


class KerrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KerrasDPM": KerrasDPM,
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

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SD15_WEIGHTS,
            controlnet=ControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, "temporalnet2"),
                torch_dtype=torch.float16,
            ),
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe._progress_bar_config = dict(disable=True)
        self.safety_checker = self.pipe.safety_checker
        self.raft = (
            raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
            .eval()
            .to("cuda")
        )

        print("Setup complete in %f" % (time.time() - st))

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for the model"),
        input_video: Path = Input(description="Input video"),
        negative_prompt: str = Input(  # FIXME
            description="Negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        strength: float = Input(
            description="How heavily stylization affects the image", default=0.7
        ),
        controlnet_scale: float = Input(
            description="Controlnet conditioning scale",
            default=1.0,
        ),
        width: int = Input(
            description="width",
            default=512,
        ),
        height: int = Input(
            description="height",
            default=512,
        ),
        batch_size: int = Input(
            description="Batch size (more = faster but more memory)",
            default=4,
        ),
        scheduler: str = Input(
            default="KerrasDPM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        num_inference_steps: int = Input(
            description="Steps to run denoising", default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
    ) -> List[Path]:
        if len(MISSING_WEIGHTS) > 0:
            raise Exception("missing weights")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            self.pipe.safety_checker = None
        else:
            self.pipe.safety_checker = self.safety_checker

        raft = (
            raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
            .eval()
            .to("cuda")
        )

        if os.path.exists("input.mov"):
            os.unlink("input.mov")
        shutil.copy(input_video, "input.mov")

        input_video, _, info = read_video(
            "input.mov", pts_unit="sec", output_format="TCHW"
        )
        input_video = input_video.div(255)

        output_video = []
        for i in trange(
            1,
            len(input_video),
            batch_size,
            desc="Diffusing...",
            unit="frame",
            unit_scale=batch_size,
        ):
            prev = resize(
                input_video[i - 1 : i - 1 + batch_size], (height, width), antialias=True
            ).to("cuda")
            curr = resize(
                input_video[i : i + batch_size], (height, width), antialias=True
            ).to("cuda")
            prev = prev[
                : curr.shape[0]
            ]  # make sure prev and curr have the same batch size (for the last batch)

            flow_img = flow_to_image(
                self.raft.forward(*raft_transform(prev, curr))[-1]
            ).div(255)
            control_img = torch.cat((prev, flow_img), dim=1)

            output, _ = self.pipe(
                prompt=[prompt] * curr.shape[0],
                image=curr,
                control_image=control_img,
                height=height,
                width=width,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=generator,
                output_type="pt",
                return_dict=False,
            )

            output_video.append(output.permute(0, 2, 3, 1).cpu())

        out_file = "output.mp4"
        if os.path.exists(out_file):
            os.remove(out_file)

        write_video(out_file, torch.cat(output_video).mul(255), fps=info["video_fps"])

        return [Path(out_file)]
