from ctransformers import AutoModelForCausalLM
import os
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("V:/mistral/", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_type="mistral", gpu_layers=50, context_length=1024)
mainPrompt = "[INST] You are a translator from file names to full descriptions.\
          You will be given the name of a texture file (example: 4K-fabric_brown-diffuse.jpg) and you must translate it into a set of tags that describe it. Tags are separated by commas, and are short, like \"realistic\". \
          (example, using 4K-fabric_brown-diffuse.jpg: A professional 4k diffuse texture of a brown fabric, extremely detailed, high quality, vibrant colors, photoscan, albedo, top down). \
          Note that there are several types of texture: \"normal\" or \"nrm\" is a normal map (color), \
          \"albedo\" or \"diffuse\" or \"col\" is an albedo map (color), \
          \"height\" or \"disp\" or \"dis\" is a height map (b&w), \
          \"ao\" is an ambient occlusion map (b&w), \
          \"rgh\" or \"roughness\" is a roughness map (b&w), \
          \"met\" or \"metallic\" is a metallic map (b&w),\
          \"bmp\" or \"bump\" is a bump map (b&w),\
          \"refl\" or \"spec\" is a specular map (color or b&w),\
          \"ems\" or \"emt\" is an emission map (color),\
          You must take this into account and structure the description based on the type of texture. For example, a roughness map SHOULD NOT HAVE \"vibrant colors\" as a tag, because it's black and white. Same thing with the height, AO, and metallic maps. \
          Tags are not sentences, however. They may only include elements that might be present in the file, such as surface imperfections or details about the texture itself (like 4K, highly detailed, etc.). \
          Use as many adjectives as possible, but do not output more than 30 words. ONLY OUTPUT THE RESULTING DESCRIPTION.\
          [/INST]\nFile name: "
fileNames = ["4K-brick_wall_2-rgh.jpg","Asphalt03_nrm.jpg","Brick03_disp.jpg","PaintedMetal03_mask.png","PavingStones07_col.png"]
#for i in range(len(fileNames)):
#    print(fileNames[i]+" is described as:")
#    print(llm(mainPrompt+fileNames[i]+"\nDescription: ",max_new_tokens=128, temperature=0.2, repetition_penalty=1.1))
rootdir = 'V:/CC0/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if(file[-3:] != "txt" and "render" not in file.lower()):
            print("file "+file+" in folder "+os.path.basename(subdir)+" is described as: ")
            description = str(llm(mainPrompt+file+"\nFolder name: "+os.path.basename(subdir)+"\nDescription: ",max_new_tokens=128, temperature=0.2, repetition_penalty=1.1, reset=True))
            if("albedo" in file.lower()  or "diffuse" in file.lower()  or "dif" in file.lower()  or "col" in file.lower()  or "color" in file.lower()):
                description = description + " colormap"
            if("normal" in file.lower()  or "nrm" in file.lower()  or "norm" in file.lower()  or "norma" in file.lower()):
                description = description + " normalmap"
            if("rough" in file.lower()  or "rgh" in file.lower()  or "roughness" in file.lower()  or "gloss" in file.lower()):
                description = description + " roughmap"
            if("ao" in file.lower()  or "ambient" in file.lower()  or "occlusion" in file.lower()  or "darkening" in file.lower()  in file.lower()):
                description = description + " ambmap"
            if("height" in file.lower() or "displacement" in file.lower() or "disp" in file.lower() or "dis" in file.lower()):
                description = description + " heighmap"
            if("met" in file.lower() or "metal" in file.lower() or "metallic" in file.lower() or "metals" in file.lower()):
                description = description + "metalmap"
            if("spc" in file.lower()  or " specular" in file.lower()  or "metallic" in file.lower()  or "glossiness" in file.lower()  or "gloss" in file.lower()):
                description = description + "specmap"
            if("refl" in file.lower()  or "reflection" in file.lower()  or "reflcolor" in file.lower()  or "speccolor" in file.lower()):
                description = description + " refmap"
            if("bump" in file.lower()  or "bmp" in file.lower()):
                description = description + " bumpmap"
            if("emt" in file.lower()  or "emis" in file.lower()):
                description = description + " emmap"
            print(description)
            file = open(os.path.join(subdir,file[:-3]+"txt"), "w")
            print("opened file")
            file.write(description)
            file.close()
            print("file closed")
