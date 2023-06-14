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
        "sample.canny.png",
        prompt="taylor swift in a mid century modern bedroom",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.both.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.scaled.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        hough_conditioning_scale=0.6,
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        canny_conditioning_scale=0.9,
        seed=42,
        steps=30,
    )


if __name__ == "__main__":
    main()
