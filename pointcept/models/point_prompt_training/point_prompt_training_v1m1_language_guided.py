"""
Point Prompt Training

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.point_prompt_training import PDNorm


# @MODELS.register_module("PPT-v1m1")
# class PointPromptTraining_(nn.Module):
#     """
#     PointPromptTraining provides Data-driven Context and enables multi-dataset training with
#     Language-driven Categorical Alignment. PDNorm is supported by SpUNet-v1m3 to adapt the
#     backbone to a specific dataset with a given dataset condition and context.
#     """

#     def __init__(
#         self,
#         backbone=None,
#         criteria=None,
#         backbone_out_channels=96,
#         context_channels=256,
#         conditions=("Structured3D", "ScanNet", "S3DIS"),
#         template="[x]",
#         clip_model="ViT-B/16",
#         # fmt: off
#         class_name=(
#             "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
#             "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
#             "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
#             "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
#             "clutter", "otherstructure", "otherfurniture", "otherprop",
#         ),
#         valid_index=(
#             (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
#             (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
#             (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
#         ),
#         # fmt: on
#         backbone_mode=False,
#     ):
#         super().__init__()
#         assert len(conditions) == len(valid_index)
#         assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
#         self.backbone = MODELS.build(backbone)
#         self.criteria = build_criteria(criteria)
#         self.conditions = conditions
#         self.valid_index = valid_index
#         self.embedding_table = nn.Embedding(len(conditions), context_channels)
#         self.backbone_mode = backbone_mode
#         if not self.backbone_mode:
#             import clip

#             clip_model, _ = clip.load(
#                 clip_model, device="cpu", download_root="./.cache/clip"
#             )
#             clip_model.requires_grad_(False)
#             class_prompt = [template.replace("[x]", name) for name in class_name]
#             class_token = clip.tokenize(class_prompt)
#             class_embedding = clip_model.encode_text(class_token)
#             class_embedding = class_embedding / class_embedding.norm(
#                 dim=-1, keepdim=True
#             )
#             self.register_buffer("class_embedding", class_embedding)
#             self.proj_head = nn.Linear(
#                 backbone_out_channels, clip_model.text_projection.shape[1]
#             )
#             self.logit_scale = clip_model.logit_scale

#     def forward(self, data_dict):
#         condition = data_dict["condition"][0]
#         assert condition in self.conditions
#         context = self.embedding_table(
#             torch.tensor(
#                 [self.conditions.index(condition)], device=data_dict["coord"].device
#             )
#         )
#         data_dict["context"] = context
#         point = self.backbone(data_dict)
#         # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
#         # TODO: remove this part after make all backbone return Point only.
#         if isinstance(point, Point):
#             feat = point.feat
#         else:
#             feat = point
#         if self.backbone_mode:
#             # PPT serve as a multi-dataset backbone when enable backbone mode
#             return feat
#         feat = self.proj_head(feat)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#         sim = (
#             feat
#             @ self.class_embedding[
#                 self.valid_index[self.conditions.index(condition)], :
#             ].t()
#         )
#         logit_scale = self.logit_scale.exp()
#         seg_logits = logit_scale * sim
#         # train
#         if self.training:
#             loss = self.criteria(seg_logits, data_dict["segment"])
#             return dict(loss=loss)
#         # eval
#         elif "segment" in data_dict.keys():
#             loss = self.criteria(seg_logits, data_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)


@MODELS.register_module("PPT-v1m3")
class PointPromptTraining(nn.Module):
    """Point Prompt Training for multi-dataset 3D scene understanding."""

    def __init__(
        self,
        backbone: Dict,
        pdnorm: Dict,
        criteria: Optional[Dict] = None,
        backbone_out_channels: int = 64,
        context_channels: int = 256,
        dataset_labels: OrderedDict[str, List[str]] = None,
        categories: List[str] = None,
        template: str = "[x]",
        clip_model: str = "ViT-B/16",
        backbone_mode: bool = False,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
    ):
        """Initialize the PointPromptTraining model."""
        super().__init__()

        self.backbone_mode = backbone_mode
        self.template = template

        # First we'll insert our new class names as required.
        self.categories = categories

        # Create PDNorm factory and inject it into backbone config
        norm_layer_factory = self.create_pdnorm_factory(pdnorm, conditions)
        self.backbone = MODELS.build(
            {**backbone, "norm_layer_factory": norm_layer_factory}
        )

        self.criteria = build_criteria(criteria)
        self.clip_model_string = clip_model
        self.backbone_out_channels = backbone_out_channels

        # Trigger setter to initialize everything
        conditions = ("Structured3D", "ScanNet", "S3DIS")
        self.conditions = conditions
        self.embedding_table = nn.Embedding(len(conditions), context_channels)

        if not self.backbone_mode:
            self.update_class_embeddings()
        self.tag_ = "PPT-v1m3"

    def create_pdnorm_factory(self, pdnorm_config: Dict, conditions: Tuple[str]):
        """Create a factory function for PDNorm layers based on the config."""
        if not pdnorm_config["use_pdnorm"]:
            return None

        def create_pd_norm_layers():
            conditions = ('ScanNet', 'S3DIS', 'Structured3D')
            bn_layer = (
                partial(
                    PDNorm,
                    norm_layer=partial(
                        nn.BatchNorm1d,
                        eps=pdnorm_config["eps"],
                        momentum=pdnorm_config["momentum"],
                        affine=pdnorm_config["affine"],
                    ),
                    conditions=conditions,
                    decouple=pdnorm_config["decouple"],
                    adaptive=pdnorm_config["adaptive"],
                )
                if pdnorm_config["bn"]
                else partial(
                    nn.BatchNorm1d,
                    eps=pdnorm_config["eps"],
                    momentum=pdnorm_config["momentum"],
                )
            )

            ln_layer = (
                partial(
                    PDNorm,
                    norm_layer=partial(
                        nn.LayerNorm,
                        eps=pdnorm_config["eps"],
                        elementwise_affine=pdnorm_config["affine"],
                    ),
                    conditions=conditions,
                    decouple=pdnorm_config["decouple"],
                    adaptive=pdnorm_config["adaptive"],
                )
                if pdnorm_config["ln"]
                else partial(nn.LayerNorm, eps=pdnorm_config["eps"])
            )

            return bn_layer, ln_layer

        return create_pd_norm_layers

    def update_class_embeddings(self) -> None:
        """Update class embeddings based on current dataset labels."""
        import clip

        clip_model, _ = clip.load(
            self.clip_model_string, device="cpu", download_root="./.cache/clip"
        )
        clip_model.requires_grad_(False)

        self.proj_head = nn.Linear(
            self.backbone_out_channels, clip_model.text_projection.shape[1]
        )
        # print(f"proj_head shape says {self.proj_head}")
        self.logit_scale = clip_model.logit_scale

        class_prompt = [self.template.replace("[x]", name) for name in self.categories]
        class_token = clip.tokenize(class_prompt)
        class_embedding = clip_model.encode_text(class_token)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)

        # Update class embedding
        # if "class_embedding" in self._buffers:
        #     del self._buffers["class_embedding"]
        self.register_buffer("class_embedding", class_embedding, persistent=True)

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        print(f"{data_dict.keys()=}")
        condition = data_dict["condition"][0]
        assert condition in self.conditions

        # Get context embedding for the current dataset
        context = self.embedding_table(
            torch.tensor(
                [self.conditions.index(condition)], device=data_dict["coord"].device
            )
        )
        data_dict["context"] = context
        # print(f"{context.shape}=")
        # raise ValueError

        # Get features from backbone
        point = self.backbone(data_dict)
        feat = point.feat if isinstance(point, Point) else point

        if self.backbone_mode:
            return feat

        # Project features and compute similarities
        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        # print(f"{feat.shape=}")

        # print(f"{self.class_embedding.shape=}")
        print(f"{feat=} {self.class_embedding=}")
        sim = feat @ self.class_embedding[:].t()
        # print(f"{sim.shape=}")

        logit_scale = self.logit_scale.exp()
        # print(f"{logit_scale=} {sim=}")
        seg_logits = logit_scale * sim
        # print(f"Segment key says: {data_dict['segment']}")
        data_dict['segment'] = data_dict['segment'] - 1
        print("Label max:", data_dict['segment'].max())
        print("Label min:", data_dict['segment'].min())


        # Compute loss or return logits based on mode
        if self.training:
            # print(f"{seg_logits=} {data_dict['segment']=}")
            loss = self.criteria(seg_logits, data_dict["segment"])
            # print(f"{loss=}")
            raise ValueError
            return dict(loss=loss)
        elif "segment" in data_dict.keys():
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)
