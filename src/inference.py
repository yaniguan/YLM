"""Single-molecule inference with a caller-supplied frozen text encoder."""
import torch
from src.data import collate_molecules


def predict_vector(model, Z, pos, question, encode_question, task=0):
    """Return a detached (3,) vector on the model device.

    encode_question(question) returns token states (T, text_dim), excluding
    padding. Coordinates must use the model's training units (v0: bohr).
    This helper does not load weights or download a text encoder.
    """
    parameter = next(model.parameters())
    sample = dict(Z=torch.as_tensor(Z, dtype=torch.long, device=parameter.device),
                  pos=torch.as_tensor(pos, dtype=parameter.dtype, device=parameter.device),
                  text=encode_question(question).to(parameter), task=task)
    batch = collate_molecules([sample])
    training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            return model(**batch)[0][0].clone()
    finally:
        model.train(training)
