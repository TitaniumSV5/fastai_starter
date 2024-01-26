import subprocess

subprocess.run(['pip', 'install', '-Uqq', 'fastai'])
subprocess.run(['pip', 'install', 'gradio==3.50'])


from fastai.vision.all import *
import gradio as gr

def which_rocket(get_y): return get_y

learn = load_learner('model.pkl')

categories = ('Falcon 9', 'Falcon Heavy', 'Starship')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)