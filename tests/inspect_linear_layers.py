
from torch import nn

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights


model = TinyInferGPT2()
load_gpt2_weights(model)

print("\nLINEAR MODULE NAMES")
print("===================")

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name)
