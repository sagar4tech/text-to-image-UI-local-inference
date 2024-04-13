from fastapi import FastAPI
import gradio as gr 

from inference import demo 

app = FastAPI()

@app.get('/')
async def root():
    return 'Gradio is running ' , 200

app = gr.mount_gradio_app(app , demo , path='/gradio')