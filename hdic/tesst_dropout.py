from keras_facenet import FaceNet
from tensorflow.keras.layers import Dropout

embedder = FaceNet()
model = embedder.model

# List all dropout layers and their dropout rates
for i, layer in enumerate(model.layers):
    if isinstance(layer, Dropout):
        print(f"{i}: {layer.name} -> rate={layer.rate}")
