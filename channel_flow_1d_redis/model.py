"""
model.py — MLP for channel flow nu_t+ correction.

Input:  [u+/20, |du+/dy+|*Re_tau/20, log(1+nu_t)/4]   3 features
Output: delta_nu_t+ (additive correction to baseline mixing length)

Features are wall-unit normalised quantities directly analogous to
RANS closure features (u/u_ref, S_ij normalised, nu_t/nu_t_ref).
"""
try:
    import torch
    import torch.nn as nn

    class ChannelCorrectionNet(nn.Module):
        def __init__(self, n_features=3, hidden=64, n_layers=4):
            super().__init__()
            layers = []
            in_dim = n_features
            for _ in range(n_layers - 1):
                layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def make_model():
        return ChannelCorrectionNet(n_features=3, hidden=64, n_layers=4)

    def channel_loss(pred, target):
        return ((pred - target) ** 2).mean()

    def export_onnx(model, output_path):
        model.eval()
        dummy = torch.zeros(1, 3)
        torch.onnx.export(model, dummy, output_path,
                          input_names=["features"],
                          output_names=["delta_nu_t"],
                          dynamic_axes={"features": {0: "batch"}, "delta_nu_t": {0: "batch"}})
        print(f"[model] exported -> {output_path}")

except ImportError:
    class ChannelCorrectionNet:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required: pip install torch")
    def make_model():
        raise ImportError("PyTorch required")
    def channel_loss(*a, **kw):
        raise ImportError("PyTorch required")
    def export_onnx(*a, **kw):
        raise ImportError("PyTorch required")
