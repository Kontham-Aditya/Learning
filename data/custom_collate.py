# custom_collate.py

import torch

def custom_collate_fn(batch):
    elem = batch[0]
    collated = {}

    for key in elem:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                collated[key] = values  # Variable-sized tensors
        else:
            collated[key] = values  # Non-tensor types (e.g., dicts)
    return collated