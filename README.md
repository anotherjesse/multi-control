# Cog Implementation of ControlNet 

This is an implementation of the [Diffusers ControlNet](https://huggingface.co/blog/controlnet) as a Cog model. [Cog](https://github.com/replicate/cog) packages machine learning models as standard containers.

First, download the pre-trained weights:

`cog run script/download_weights`

Then, you can run predictions:

`cog predict -i image=@monkey.png -i prompt="monkey scuba diving" -i structure='canny'`
