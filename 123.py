import tensorflow as tf
import numpy as np
import json

# Load the .h5 model
model = tf.keras.models.load_model('fashion_mnist.h5')

# Extract architecture
arch = []
for layer in model.layers:
    layer_info = {
        'name': layer.name,
        'type': layer.__class__.__name__,
        'config': {'activation': layer.activation.__name__ if hasattr(layer, 'activation') else None},
        'weights': [f'{layer.name}/kernel:0', f'{layer.name}/bias:0'] if layer.__class__.__name__ == 'Dense' else []
    }
    arch.append(layer_info)

# Save architecture to json
with open('fashion_mnist.json', 'w') as f:
    json.dump(arch, f, indent=2)

# Extract and save weights
weights = {}
for layer in model.layers:
    if layer.weights:
        for w in layer.weights:
            weights[w.name] = w.numpy()
np.savez('fashion_mnist.npz', **weights)
