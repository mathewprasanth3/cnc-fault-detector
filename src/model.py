import torch
import torch.nn as nn


class CNCFaultDetector(nn.Module):
    def __init__(self, input_size=5, hidden_sizes=[64, 32, 16], dropout_rate=0.3):
        super(CNCFaultDetector, self).__init__()

        # building layers dynamically so we can experiment with architecture easily
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # single output neuron with sigmoid gives us a probability between 0 and 1
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


if __name__ == "__main__":
    model = CNCFaultDetector()
    print(model)

    # sanity check — pass a fake batch of 8 samples through to confirm shapes work
    dummy_input = torch.randn(8, 5)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output.detach().numpy().round(3)}")