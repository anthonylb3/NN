

# app.py
import torch
import gradio as gr
from PIL import Image, ImageOps
from torchvision import transforms

from model import MLP

# Labels van FashionMNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

ckpt = torch.load("best_model.pth", map_location="cpu")
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

print("Aantal keys:", len(sd))
print("Eerste 30 keys:")
for k in list(sd.keys())[:30]:
    print(k, sd[k].shape if hasattr(sd[k], "shape") else type(sd[k]))

# Preprocessing: maakt van een foto iets dat lijkt op FashionMNIST input
# Normalize((0.5,), (0.5,)) => schaalt pixelwaarden naar ongeveer [-1, 1]
PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(checkpoint_path="best_model.pth"):
    model = MLP(n_inputs=[3072], 
            n_hidden=[128, 256, 256, 256, 256, 128, 128, 128, 128], 
            n_classes=[10], use_batch_norm=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Jij gebruikte eerder: ckpt["model_state_dict"]
    # Soms is het direct een state_dict. We ondersteunen beide:
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model

MODEL = load_model('best_model.pth')

def predict(image: Image.Image, invert: bool):
    """
    image: PIL image uit Gradio
    invert: als True -> kleuren omkeren (kan helpen als je foto "omgekeerd" is t.o.v. FashionMNIST)
    """
    if image is None:
        return "No image", {}

    # Zorg voor RGB (consistent), daarna eventueel invert
    img = image.convert("RGB")
    if invert:
        img = ImageOps.invert(img)

    # Preprocess naar tensor: (1, 1, 28, 28)
    x = PREPROCESS(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # shape (10,)
        pred_idx = int(torch.argmax(probs).item())

    # Maak output dict voor Gradio Label (class->prob)
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, prob_dict


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload image"),
        gr.Checkbox(value=False, label="Invert (try if predictions are weird)"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=3, label="Top-3 probabilities"),
    ],
    title="FashionMNIST Classifier (Gradio)",
    description=(
        "Upload een foto. De app zet 'm om naar 28×28 grayscale zoals FashionMNIST en voorspelt de klasse. "
        "Let op: echte foto's zijn vaak anders dan FashionMNIST, dus resultaten kunnen wisselen."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
