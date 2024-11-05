from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import gc
import torch
import gradio as gr

#initialize constants
dir = os.path.join(os.getcwd(),"output")
    
sumDir = dir

generationDir = dir
if not os.path.exists(generationDir) or not os.path.isdir(generationDir):
    os.mkdir(generationDir)
if not os.path.exists(sumDir) or not os.path.isdir(sumDir):
    os.mkdir(sumDir)
    
LCMMode = False
textData = []
mapPrompts = []
global realDesc
textPipeline = ""
mapPrompts = []

from ctransformers import AutoModelForCausalLM

mapTypes = ["height","albedo","roughness","normal","specular","ambient occlusion"]
    
specialStrings = ["black and white, small height variations, heighmap",
                  "VAR2, colormap, muted realistic colors",
                  "black and white, small roughness variations, roughmap",
                  "normal map",
                  "black and white, specmap",
                  "black and white, ambmap"]
    
stockBase = "An 8k PBR texture of bricks, high resolution, detailed, realistic, dirt, architectural element, building material"
stockPrompts = ["an 8k high resolution height map of bricks, highly detailed, realistic surface imperfections, photoscan, grayscale, heighmap",
                "an 8k albedo texture of bricks, highly detailed, realistic, realistic colors, albedo, top down colormap", 
                "an 8k roughness texture of bricks, highly detailed, realistic surface imperfections, roughness variation, high quality, black and white, photoscan, albedo, top down, roughmap", 
                "an 8k normal map of a brick wall, high resolution, realistic surface imperfections and details, normalmap", 
                "an 8k specular map of bricks, highly detailed, realistic texture, black and white, realistic imperfections, top down, specmap", 
                "an 8k ambient occlusion texture of bricks, highly detailed, realistic surface imperfections, photoscan, black and white, ambmap"]

translatePrompt = "Turn this description of a material into a description of it's {} map. \
Use roughly the same format as the original description, with a main body consisting of a description of the material and several tags describing minute details, but change it so that it describes the {} map. Remember that height, roughness, and AO maps are black-and-white, whilst albedo maps are in color. Normal maps cannot include information about the material color (as they, just like height maps, describe relief), but they do use color to represent the fine height variations. Modify tags appropriately.\
Example:\n"
secondPart = "It's recommended that you translate most of the tags from the original prompt into the {} prompt, however, you must use tags relating to the material being translated, rather than just copying the ones in the example.\n\
YOU ARE REQUIRED TO INCLUDE THE MAIN BODY, EXPLAINING WHAT THE MATERIAL IS. \n\
YOU MUST FOLLOW THIS TEMPLATE. DO NOT OUTPUT ANYTHING ELSE.\n"

negativePrompts = ["weird, ugly, low quality, messed up, unrealistic, VAR1, blurry, low resolution",
                   "weird, ugly, low quality, messed up, unrealistic, VAR1, blurry, low resolution, garish colors, very vibrant",
                   "weird, ugly, low quality, messed up, unrealistic, VAR1",
                   "colormap",
                   "weird, ugly, low quality, messed up, unrealistic, VAR1",
                   "weird, ugly, low quality, messed up, unrealistic, VAR1"]

output_descriptions = []

def flush():
  gc.collect()
  torch.cuda.empty_cache()

def rescale(img):
    ar = np.array(img)
    mn = np.linalg.norm(np.min(ar))
    mx = np.linalg.norm(np.max(ar))
    norm = (ar - mn) * (1.0 / (mx - mn))
    return norm

def classifyText(inputString):
    selected_model = "madhurjindal/autonlp-Gibberish-Detector-492513457"
    classifier = pipeline("text-classification", model=selected_model)
    #userPrompt = input("Input texture prompt: ")
    classification = classifier(inputString)
    #del classifier
    #flush()
    return classification
global quantized
quantized = False
def generatePBRPrompts(userPrompt, use4bitModel):
    print("Text accepted.")
    model_name_or_path = str(os.path.join(os.getcwd(),"gptq")).replace('\\','/')+"/"
    print("Loading model...")
    global quantized
    quantized = use4bitModel
    
    if(use4bitModel):
        global llm
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=50, context_length=1024, max_new_tokens=64)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="cuda",
                                                trust_remote_code=False).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, skip_special_tokens=True)

    
    print("Model loaded!")
    print("Preparing summarization...")
    
    system = "You will be given a short description or prompt to generate a material. You must translate that description or prompt into a detailed description of the material. Example:\
    \nINPUT: Generate something that looks like a rocky mountain\nOUTPUT: An 8k professional PBR texture of a rock, high quality, extremely detailed, small variations, dirt, realistic\n\
    Every texture's description consists of two sections: the main body and the tags. The main body defines what the material is, whilst the tags define minute details. For example, in the previous input-output pair, \"An 8k professional PBR texture of a rock\" is the main body, whilst everything else is the tags. The main body MUST include several nouns, being the description of what is located in the texture. In the example, \"rock\" is the description of what the texture contains. YOU MUST INCLUDE THIS. The main body also MUST include the words \"PBR texture\" because your objective is to generate descriptions for textures. Do not include things like \"3D model\" or \"illustration\" or anything that isn't related to PBR textures. YOU ARE REQUIRED TO INCLUDE THE MAIN BODY. Tags MUST include things related to the texture itself, such as detail. Tags are typically 2 to 4 words in length. Do not use more than that. You must put it at least THREE tags, but YOU ARE NOT ALLOWED TO USE MORE THAN TEN TAGS. You are REQUIRED to write both parts and follow this template.\
    \nFollow the given template to turn any description or text into a texture description.\
    Make sure to only output the texture description, and nothing else."
    model_prompt = f'''<s>[INST] {system} [/INST]\nINPUT: {userPrompt}\nOUTPUT: '''
    summarizationPrompt = "You will be given a short description of a material, and you must summarize it to only use two words. Output only the two words that best describe the prompt. Example: 'a high quality texture of cooked rice grains' => 'cooked rice'. REPEAT ACCORDINGLY WITH THE GIVEN PROMPT.\nINPUT:{}\nOUTPUT:".format(userPrompt)

    if(use4bitModel):
        summarizedName = str(llm(summarizationPrompt, max_new_tokens=10)).strip('\n') #type: ignore
    else:
        global textPipeline
        textPipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, temperature=0.2, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=64, repetition_penalty=1.1) #type: ignore
        summarizedName = textPipeline(summarizationPrompt, max_new_tokens=10)[0]["generated_text"][len(summarizationPrompt):].strip('\n') #type: ignore


    
    
    
    print("Summarized prompt:",summarizedName)
    
    print("To be made:",os.path.join(os.getcwd(),summarizedName))
    
    
    
    dir = os.path.join(os.getcwd(),"output")
    
    sumDir = os.path.join(dir,summarizedName)
    
    if not os.path.exists(dir) or not os.path.isdir(dir):
        os.mkdir(dir)
    
    #generationDir = os.path.join(dir,summarizedName)
    generationDir = dir
    if not os.path.exists(generationDir) or not os.path.isdir(generationDir):
        os.mkdir(generationDir)
    if not os.path.exists(sumDir) or not os.path.isdir(sumDir):
        os.mkdir(sumDir)
    
    realDesc = ""
    global output_descriptions
    
    output_descriptions = []
    for i in range(4):
        if(not use4bitModel):
            result = textPipeline(model_prompt) #type: ignore
        
            print(f'''#{i+1}:''',result[0]["generated_text"][len(model_prompt):].strip('\n')) # type: ignore
            realDesc = realDesc + (f'''#{i+1}: '''+result[0]["generated_text"][len(model_prompt):].strip('\n')+"\n") # type: ignore
            output_descriptions.append(result[0]["generated_text"][len(model_prompt):].strip('\n')) # type: ignore
        else:
            result = llm(model_prompt)
            print(f'''#{i+1}:''',str(result).strip('\n')) # type: ignore
            realDesc = realDesc + (f'''#{i+1}: '''+str(result).strip('\n')+"\n") # type: ignore
            output_descriptions.append(str(result).strip('\n')) # type: ignore
        
    #choice = input("Which description would you like to generate textures for? (1-4): ") # type: ignore
    #print(realDesc)
    return "Which description would you like to generate textures for? (1-4):\n"+realDesc


def generateText(choice):
    global output_descriptions
    print(output_descriptions)
    choice = output_descriptions[min(max(int(choice)-1,0),len(output_descriptions))]
    print("You've chosen:",choice)
    
    global mapPrompts
    mapPrompts = []
    global textPipeline
    global llm
    global quantized
    for i in range(6):
        if(not quantized):
            finalPrompt = f'''<s>[INST] {translatePrompt.format(mapTypes[i],mapTypes[i])}INPUT PBR DESCRIPTION:{stockBase}\nOUTPUT {mapTypes[i].upper} MAP DESCIPTION: {stockPrompts[i]}\n{secondPart.format(mapTypes[i])}[/INST]\nINPUT PBR DESCRIPTION:{choice}\nOUTPUT {mapTypes[i].upper} MAP DESCRIPTION: '''
            mapPrompts.append(textPipeline(finalPrompt)[0]["generated_text"][len(finalPrompt):].strip('\n')+","+specialStrings[i]) # type: ignore
            print(mapTypes[i],"map prompt:",mapPrompts[i])
        else:
            finalPrompt = f'''<s>[INST] {translatePrompt.format(mapTypes[i],mapTypes[i])}INPUT PBR DESCRIPTION:{stockBase}\nOUTPUT {mapTypes[i].upper} MAP DESCIPTION: {stockPrompts[i]}\n{secondPart.format(mapTypes[i])}[/INST]\nINPUT PBR DESCRIPTION:{choice}\nOUTPUT {mapTypes[i].upper} MAP DESCRIPTION: '''
            llmdata = str(llm(finalPrompt))
            mapPrompts.append(llmdata.strip('\n')+","+specialStrings[i]) # type: ignore
            print(mapTypes[i],"map prompt:",mapPrompts[i])
    
    
    #del textPipeline
    #del tokenizer
    #del model
    #flush()
    return mapPrompts


def generateImages(SampleSteps,guidanceScale):
    generator = torch.Generator(device="cuda")#.manual_seed(69420) #nice
    
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    #pipelineDiff = AutoPipelineForText2Image.from_pretrained(
    #    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, vae=vae
    #)

    #pipelineDiff.load_lora_weights("3drous-last-step00008800.safetensors")
    
    #pipelineDiff.scheduler = DPMSolverMultistepScheduler.from_config(pipelineDiff.scheduler.config, use_karras_sigmas=True) #type: ignore
    #pipelineDiff.enable_xformers_memory_efficient_attention()
    #pipelineDiff.to("cuda")
    
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    
    heightmap = 0
    
    controlnet_conditioning_scale = 0.7
    
    #generate base height image
    
    global mapPrompts
    
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True) #type: ignore
    #pipe.load_lora_weights("3drous-last-step00008800.safetensors") #development name for the thing (tm). Or at least the 3D model.
    pipe.load_lora_weights("texture-synthesis-topdown-base-condensed.safetensors")
    pipe.controlnet.to(memory_format=torch.channels_last)
    #tomesd.apply_patch(pipe, ratio=0.5)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    
    
    image = pipe(prompt=mapPrompts[0], negative_prompt=negativePrompts[0],controlnet_conditioning_scale=controlnet_conditioning_scale, image=Image.fromarray((np.zeros((1024,1024,3)) * 255).astype(np.uint8)), num_inference_steps = SampleSteps, generator=generator, guidance_scale = guidanceScale).images[0] #type: ignore
    image = rescale(image)
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(str(os.path.join(generationDir,"height.png")))
    image.save(str(os.path.join(sumDir,"height.png")))
    image = cv2.Canny((np.array(image) * 255).astype(np.uint8), 20, 70) #type: ignore
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    heightmap = Image.fromarray(image)
    
    #generate the rest
    imageArray = []
    for i in range(5):
        #if(LCMMode):
        #    image = pipe(prompt=mapPrompts[i+1], negative_prompt=negativePrompts[i+1],controlnet_conditioning_scale=controlnet_conditioning_scale, image=heightmap, num_inference_steps = 8, generator=generator, guidance_scale = 2.2) #type: ignore
        #else:
        image = pipe(prompt=mapPrompts[i+1], negative_prompt=negativePrompts[i+1],controlnet_conditioning_scale=controlnet_conditioning_scale, image=heightmap, num_inference_steps = SampleSteps, generator=generator, guidance_scale = guidanceScale) #type: ignore
        
        
        image.images[0].save(str(os.path.join(generationDir,mapTypes[i+1]+".png"))) #type: ignore
        image.images[0].save(str(os.path.join(sumDir,mapTypes[i+1]+".png"))) #type: ignore
        imageArray.append(image.images[0]) #type: ignore

    #for i in range(5):
    #    (images.images[i]).save(str(os.path.join(generationDir,mapTypes[i+1]+".png")))
    
    #image = try_numpy(image)
    #del pipe #type: ignore
    #flush()
    return imageArray



custom_css = """
    #button {
        flex-grow: 1;
    }
    #output .tabs {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    #output .tabitem[style="display: block;"] {
        flex-grow: 1;
        display: flex !important;
    }
    #output .gap {
        flex-grow: 1;
    }
    #output .form {
        flex-grow: 1 !important;
    }
    #output .form > :last-child{
        flex-grow: 1;
    }
"""

firstStageOutputText = ""
with gr.Blocks(css=custom_css) as demo:
    with gr.Tab("txt2txt2img"):
        with gr.Row(scale=4).style(equal_height=True):
            with gr.Column(scale=4):
                firstStageInputText = gr.Textbox(label="Base prompt")
                FourBitMode = gr.Checkbox(label="Use 4bit mode", info="WARNING: This may cause a slight quality loss, in exchange for a large speed gain.", value=True)
            with gr.Column(scale=4):
                firstStageOutputText = gr.Textbox(label="Expanded prompt")
                firstStageLaunchButton = gr.Button("Expand Prompt",variant="primary") 
        with gr.Row(scale=4,variant="panel").style(equal_height=True):
            with gr.Column():
                secondStageChoice = gr.Radio([1, 2, 3, 4], label="Choose one prompt for generation:") #type: ignore
                
                secondStageOutputText = gr.Textbox(label="Texture map prompts:")
                secondStageLaunchButton = gr.Button("Generate Prompts",variant="primary",elem_id="button")
            with gr.Column(elem_id="output"):
                #with gr.Column(scale=1, min_width=300):
                gallery = gr.Gallery(label="Generated images", show_label=False, columns=3, rows=2)
                btn = gr.Button("Generate Images", scale=0,variant="primary")
                
        SampleSteps = gr.Slider(1, 70, value=20, label="Sampling steps")
        CFGScale = gr.Slider(1.0, 16.0, value=7.0, label="CFG scale")

        btn.click(generateImages, [SampleSteps,CFGScale], gallery)
            

        
        firstStageLaunchButton.click(fn=generatePBRPrompts, inputs=[firstStageInputText, FourBitMode], outputs=firstStageOutputText)
        
        secondStageLaunchButton.click(fn=generateText, inputs=secondStageChoice, outputs=secondStageOutputText)
        
        #gr.Interface(fn=generatePBRPrompts, inputs=["text", FourBitMode], outputs="text")
        
        #gr.Interface(fn=generateText, inputs="text", outputs="text")
        
    with gr.Tab("img2img"):
        FourBitMode = gr.Checkbox(label="Test", info="Still unfinished", value=False)
        #guess who didn't finish this part before moving on to other projects

if __name__ == "__main__":
    demo.launch()