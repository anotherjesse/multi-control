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
        print(data['logs'])
        sys.exit(1)
 
    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.depth.png",
        prompt="painting of farmer by van gogh",
        structure="depth",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42,
        steps=20,
    )
    gen(
        "sample.canny.png",
        prompt="painting of farmer by van gogh",
        structure="canny",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42,
        steps=20,
    )
    gen(
        "sample.normal.png",
        prompt="painting of farmer by van gogh",
        structure="normal",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42,
        steps=20,
    )


if __name__ == "__main__":
    main()
