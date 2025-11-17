# print_dropout.py
import torch
from facenet_pytorch import InceptionResnetV1

def find_dropout_layers(model):
    drops = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            drops.append((name, m.p))
    return drops

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    drops = find_dropout_layers(model)
    if not drops:
        print("No torch.nn.Dropout layers found.")
    else:
        print("Found Dropout layers:")
        for name, p in drops:
            print(f"  - {name}: p={p}")
