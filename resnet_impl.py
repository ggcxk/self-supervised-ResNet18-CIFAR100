import torch.nn as nn
import torchvision.models as models


class get_resnet18(nn.Module):

    def __init__(self, out_dim):
        super(get_resnet18, self).__init__()
        self.backbone = self._get_basemodel(out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, out_dim):
        return models.resnet18(pretrained=False, num_classes=out_dim)

    def forward(self, x):
        return self.backbone(x)
