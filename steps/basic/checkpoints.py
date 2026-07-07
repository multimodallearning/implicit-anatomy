import os
import urllib.parse

import torch
from torch.utils import model_zoo


class CheckpointIO:
    def __init__(self, checkpoint_dir="./checkpoints", device="cpu", **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        os.makedirs(checkpoint_dir, exist_ok=True)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, filename, **scalars):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        state = dict(scalars)

        for name, module in self.module_dict.items():
            state[name] = module.state_dict()

        torch.save(state, filename)

    def load(self, filename):
        if is_url(filename):
            state = model_zoo.load_url(filename, progress=True)
            return self.parse_state_dict(state)

        return self.load_file(filename)

    def load_file(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Checkpoint not found: {filename}"
            )

        print(f"Loading checkpoint: {filename}")
        state = torch.load(filename, map_location=self.device)

        return self.parse_state_dict(state)

    def parse_state_dict(self, state):
        for name, module in self.module_dict.items():
            if name in state:
                module.load_state_dict(state[name])
            else:
                print(f"Warning: '{name}' missing from checkpoint")

        return {
            name: value
            for name, value in state.items()
            if name not in self.module_dict
        }


def is_url(path):
    scheme = urllib.parse.urlparse(path).scheme
    return scheme in ("http", "https")