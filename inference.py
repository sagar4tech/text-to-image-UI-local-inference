####   text to Image  on Gradio UI 

from diffusers import DiffusionPipeline
import gradio as gr

# Provide the file path to your locally stored model file
model_file_path = "/model/file/path/to/model/directory/"

# Provide the url to model id
url = "runwayml/stable-diffusion-v1-5"

# Load the safetensor model 
pipe = DiffusionPipeline.from_pretrained(url, torch_dtype=torch.float16)
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
