####   text to Image  on Gradio UI 

from diffusers import StableDiffusionPipeline
import gradio as gr

# Provide the file path to your locally stored model file
model_file_path = "/Users/sagar/Desktop/AI-project/gen-ai-inference/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"

# Load the safetensor model 
pipe = StableDiffusionPipeline.from_single_file(model_file_path)
pipe = pipe.to("mps")

def predict(text):
    # Ensure pipe(text) returns the correct output format
    generated_image = pipe(text).images[0]
    return generated_image



demo = gr.Interface(
    fn=predict,
    inputs='text',
    outputs='image',
)

demo.launch()
