# sdxl-texture-synthesis
An old and very bad web UI for running my texture synthesis LoRA, plus some tools I used to make it.

You'll need the transformers, diffusers, PIL, numpy, os, cv2, gc, torch, and gradio packages for this thing to work. Probably. Although at this point I barely even remember.
I'd make a requirements file but I'm too lazy.

This whole codebase is utterly terrible and doesn't represent the way I write code anymore. Consider this a historical artifact of sorts; a time capsule.

Anyways, mainTextureGen.py is the thing that generates textures. Only the txt2txt2img mode works, where you enter a prompt that then gets parsed by an LLM and turned into multiple prompts for all the different map types, and then those textures get generated.

gpt.py is the script I used to label the texture data. It uses an LLM to turn filenames into texture descriptions. It's not terribly useful considering the sheer inefficiency of it all and the existance of models better than the original mistral 7b (sep-dec 2024 my beloved), but it does provide some extra info on how the model was trained. I guess.

Have fun!
