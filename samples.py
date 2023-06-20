import base64
import requests
import sys
import os


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        os.remove(output_fn)

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
        "ada_lily.mp4",
        prompt="kids in space",
        seed=42,
        input_video="https://storage.googleapis.com/replicant-misc/sample.mov",
        width=640,
        height=360,
        disable_safety_check=True,
    )


if __name__ == "__main__":
    main()
