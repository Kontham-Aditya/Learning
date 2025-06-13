# custom_collate.py
import torch

def custom_collate_fn(batch):
    elem = batch[0]
    collated = {}

    for key in elem:
        values = [d[key] for d in batch]
        # Stack only if all values are same-sized tensors
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                # Likely variable-sized (e.g., point clouds), keep as list
                collated[key] = values
        else:
            collated[key] = values  # Non-tensor types (dict, str, etc.)
    return collated