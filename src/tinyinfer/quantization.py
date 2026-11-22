
import torch
import torch.nn as nn
from torch.ao.quantization import per_channel_dynamic_qconfig



def quantize_model_dynamic(model):
    """
    Per-channel dynamic INT8 quantization.

    Keeps lm_head and MLP output projections in FP32
    to reduce quantization error.
    """
    return quantize_selective(
        model,
        should_quantize=lambda name: (
            name != "lm_head"
            and not name.endswith("mlp.proj")
        )
    )


def quantize_selective(model, should_quantize):
    """
    Dynamically quantize selected Linear layers using
    per-channel weight quantization.

    should_quantize receives each module's name.
    """
    model.eval()

    qconfig_spec = {
        name: per_channel_dynamic_qconfig
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and should_quantize(name)
    }

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8
    )

    return quantized_model
