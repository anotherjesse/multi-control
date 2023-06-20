import base64
import requests
import sys
import os


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        print("Skipping", output_fn)
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        # sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.txt2img.png",
        prompt="taylor swift in a mid century modern bedroom",
        seed=42,
        num_inference_steps=30,
        scheduler="KerrasDPM",
    )
    gen(
        "sample.canny.png",
        prompt="taylor swift in a mid century modern bedroom",
        scheduler="KerrasDPM",
        controlnet_1="canny",
        controlnet_1_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.qr.png",
        prompt="A film still of a kraken, reconciliation, 8mm film, traditional color grading, cinemascope, set in 1878",
        scheduler="KerrasDPM",
        controlnet_1="qr",
        controlnet_1_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        controlnet_1_conditioning_scale=1.5,
        seed=42,
        num_inference_steps=50,
    )
    gen(
        "sample.canny.guess.png",
        prompt="",
                scheduler="KerrasDPM",
controlnet_1="canny",
        controlnet_1_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.hough.png",
        scheduler="KerrasDPM",
        prompt="taylor swift in a mid century modern bedroom",
        controlnet_1="hough",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.hough.guess.png",
        prompt="",
        controlnet_1="hough",
        scheduler="KerrasDPM",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.normal.png",
        prompt="",
        scheduler="KerrasDPM",
        controlnet_1="normal",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.depth.png",
        prompt="",
        scheduler="KerrasDPM",
        controlnet_1="depth",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.two.png",
        prompt="taylor swift in a mid century modern bedroom",
        controlnet_1="hough",
        scheduler="KerrasDPM",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        controlnet_2="canny",
        controlnet_2_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.both.guess.png",
        prompt="",
        controlnet_1="hough",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        scheduler="KerrasDPM",
        controlnet_2="canny",
        controlnet_2_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.scaled.png",
        scheduler="KerrasDPM",
        prompt="taylor swift in a mid century modern bedroom",
        controlnet_1="hough",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        controlnet_1_conditioning_scale=0.6,
        controlnet_2="canny",
        controlnet_2_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        controlnet_2_conditioning_scale=0.8,
        seed=42,
        num_inference_steps=30,
    )
    gen(
        "sample.seg.png",
        prompt="modern bedroom with plants",
        controlnet_1="seg",
        scheduler="KerrasDPM",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
    )
    gen(
        "sample.hed.png",
        prompt="modern bedroom with plants",
        controlnet_1="hed",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        scheduler="KerrasDPM",
        seed=42,
    )
    gen(
        "sample.pose.png",
        prompt="a man in a suit by van gogh",
        controlnet_1="pose",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        scheduler="KerrasDPM",
        seed=42,
    )
    gen(
        "sample.scribble.png",
        prompt="painting of cjw by van gogh",
        controlnet_1="scribble",
        scheduler="KerrasDPM",
        controlnet_1_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_vermeer_scribble.png",
        seed=42,
    )



if __name__ == "__main__":
    main()
