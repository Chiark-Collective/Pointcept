import torch.nn as nn
import torch

from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.builder import MODULES


@MODULES.register_module()
class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point

    def _copy_norm_params(self, source_norm, target_norm):
        with torch.no_grad():
            if hasattr(source_norm, 'weight'):
                target_norm.weight.copy_(source_norm.weight)
            if hasattr(source_norm, 'bias'):
                target_norm.bias.copy_(source_norm.bias)
            if hasattr(source_norm, 'running_mean'):
                target_norm.running_mean.copy_(source_norm.running_mean)
            if hasattr(source_norm, 'running_var'):
                target_norm.running_var.copy_(source_norm.running_var)
