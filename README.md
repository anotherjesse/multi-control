# Cog Implementation of ControlNet 

This is an implementation of the [Diffusers ControlNet](https://huggingface.co/blog/controlnet) as a Cog model. [Cog](https://github.com/replicate/cog) packages machine learning models as standard containers.

First, download the pre-trained weights:

`cog run script/download_weights`

Then, you can run predictions:

`cog predict -i image=@monkey.png -i prompt="monkey scuba diving" -i structure='canny'`

## Issues

- [ ] support aspect ratio from image (currently it is resized to a square?)
- [ ] safety results aren't checked (resulting in a black image?)
- [ ] ability to return processed control image(s)
- [ ] ability to send pre-processed control image(s)
- [ ] support for multiple control nets / images
- [ ] support for controlnet guidance scale