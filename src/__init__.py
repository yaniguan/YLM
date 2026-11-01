"""YLM models and utilities.

Older e3nn releases serialize Python slices in their bundled Wigner constants.
Allow only that harmless builtin, preserving PyTorch's weights-only loader.
Import YLM before importing e3nn directly when using those releases.
"""
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
