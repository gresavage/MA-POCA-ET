from mapoca.torch_utils import nn, torch
from mapoca.trainers.torch.layers import linear_layer


class ValueHeads(nn.Module):
    def __init__(self, stream_names: list[str], input_size: int, output_size: int = 1):
        super().__init__()
        self.stream_names = stream_names
        value_heads = {}

        for name in stream_names:
            value = linear_layer(input_size, output_size)
            value_heads[name] = value
        self.value_heads = nn.ModuleDict(value_heads)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        value_outputs = {}
        for stream_name, head in self.value_heads.items():
            value_outputs[stream_name] = head(hidden).squeeze(-1)
        return value_outputs
