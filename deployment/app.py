from fastai.vision.all import load_learner, PILImage
import gradio as gr

pitha_labels = [
 'Chit ruti Pitha',
 'Pata Pitha (Leaf-shaped Pitha)',
 'atikka pitha', 'bhapa pitha',
 'bibikhana pitha',
 'binni chaler pitha',
 'chitoi pitha',
 'choi pitha',
 'dim shundori pitha',
 'dudh chitoi pitha',
 'dudh puli pitha',
 'fuljhuri pitha',
 'jhinuk pitha',
 'khejur pitha',
 'mera pitha',
 'nokshi pitha',
 'patisapta pitha',
 'puli pitha',
 'semai pitha',
 'teler pitha'
]

model = load_learner("models/pitha-recognizer-v2.pkl")
learn=learn.export("pitha-recognizer-v2-py310.pkl")

def recognize_image(image):
    img = PILImage.create(image)
    pred, idx, probs = model.predict(img)
    return dict(zip(pitha_labels, map(float, probs)))

iface = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(),
    examples=[
        "test images/unknown_00.jpg",
        "test images/unknown_01.jpg",
        "test images/unknown_02.jpg",
        "test images/unknown_03.jpg",
    ]
)

iface.launch(share=True, ssr_mode=False)
