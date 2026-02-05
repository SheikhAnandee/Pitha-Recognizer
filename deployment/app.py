from fastai.vision.all import load_learner, PILImage
import gradio as gr

pitha_labels = [
 'Chit ruti Pitha',
 'Pata Pitha (Leaf-shaped Pitha)',
 'Atikka Pitha', 'bhapa Pitha',
 'Bibikhana Pitha',
 'Binni Chaler Pitha',
 'Chitoi Pitha',
 'Choi Pitha',
 'Dim Shundori Pitha',
 'Dudh Chitoi Pitha',
 'Dudh Puli Pitha',
 'Fuljhuri Pitha',
 'Jhinuk Pitha',
 'Khejur Pitha',
 'Mera Pitha',
 'Nokshi Pitha',
 'Patisapta Pitha',
 'Puli Pitha',
 'Semai Pitha',
 'Teler Pitha'
]

model = load_learner("models/Pitha-recognizer-v2.pkl")
learn=learn.export("Pitha-recognizer-v2-py310.pkl")

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
