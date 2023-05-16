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
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.depth.png",
        structure="depth",
        prompt="taylor swift, best quality, extremely detailed",
        image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.canny.png",
        prompt="taylor swift, best quality, extremely detailed",
        structure="canny",
        image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=20,
    )
    gen(
        "sample.normal.png",
        structure="normal",
        prompt="taylor swift, best quality, extremely detailed",
        image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.seg.png",
        structure="seg",
        prompt="mid century modern bedroom",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.hed.png",
        structure="hed",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/bird_512x512.png",
        prompt="rainbow bird, best quality, extremely detailed",
        seed=42,
        steps=30,
    )
    gen(
        "sample.pose.png",
        structure="pose",
        image="https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg",
        prompt="farmer yoga pose, best quality, extremely detailed",
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.png",
        structure="hough",
        prompt="mid century modern bedroom",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        steps=30,
    )


if __name__ == "__main__":
    main()
