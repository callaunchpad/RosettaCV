"""
Array of models useful for meta pseudo labels and its augmentations.
"""
from copy import deepcopy

class MovingAverage(nn.Module):
    def __init__(self, base_model: nn.Module, decay_rate: float, device):
        self.avg_model = deepcopy(base_model)
        self.decay_rate = decay_rate
        self.device = device
        if self.device is not None:
            self.avg_model.to(self.device)

    def forward(self, x):
        return self.avg_model(x)

    def update(self, new_model):
        # copy over all values with ema
        with torch.no_grad():
            for old_param, new_param in zip(self.avg_model.parameters(), self.new_model.parameters()):
                if self.device is not None:
                    new_param.to(self.device)
                old_param._copy(self.decay_rate*old_param + (1-self.decay_rate)*new_param)
            for old_buffer, new_buffer in zip(self.avg_model.buffers(), self.new_model.buffers()):
                if self.device is not None:
                    new_buffer.to(self.device)
                old_buffer._copy(self.decay_rate*old_buffer + (1-self.decay_rate)*new_buffer)

    def state_dict(self):
        return self.avg_model.state_dict()

    def load_state_dict(self, state_dict):
        self.avg_model.load_state_dict(state_dict)
