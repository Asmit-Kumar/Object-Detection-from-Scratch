"""Two-scale YOLOv10-style dual assignment on the existing auxiliary architecture.

Main predictions use one-to-one supervision; the smaller auxiliary neck uses
one-to-many supervision. This adopts YOLOv10's training strategy, not its full
CIB/PSA backbone. The existing direct LTRB regression is retained.
"""

import torch
from torch.nn import functional as F

from models.auxiliary_network_detector import AuxiliaryObjectDetector


class YOLOv10StyleObjectDetector(AuxiliaryObjectDetector):
    end_to_end = True

    def __init__(self, *args, inference_only: bool = False, max_detections: int = 300, **kwargs):
        if not isinstance(max_detections, int) or max_detections < 1:
            raise ValueError("max_detections must be a positive integer")
        super().__init__(*args, **kwargs)
        self.inference_only = inference_only
        self.max_detections = max_detections
        if inference_only:
            del self.auxiliary_network

    def forward(self, x: torch.Tensor):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        p3_backbone = self.stage3(self.down2(x))
        p4_backbone = self.sppf(self.stage4(self.down3(p3_backbone)))

        # As in YOLOv10, one-to-many supervision trains the shared features.
        # Our branches share the backbone rather than the neck, so detach here.
        p3_input = p3_backbone.detach() if self.training and not self.inference_only else p3_backbone
        p4_input = p4_backbone.detach() if self.training and not self.inference_only else p4_backbone
        p3 = self.top_down(torch.cat((
            p3_input,
            F.interpolate(self.reduce_p4(p4_input), size=p3_input.shape[-2:], mode="nearest"),
        ), dim=1))
        p4 = self.bottom_up(torch.cat((p4_input, self.down_p3(p3)), dim=1))
        box_p3, cls_p3 = self.head_p3(p3)
        box_p4, cls_p4 = self.head_p4(p4)
        main = {
            28: {"cls_logits": cls_p3, "reg_ltrb": self.scale_p3(box_p3) * self.STRIDES[28]},
            14: {"cls_logits": cls_p4, "reg_ltrb": self.scale_p4(box_p4) * self.STRIDES[14]},
        }
        if self.training and not self.inference_only:
            return {"main": main, "aux": self.auxiliary_network(p3_backbone, p4_backbone)}
        return main

    def to_inference(self):
        main = type(self)(
            channels=list(self.main_preset.channels),
            blocks=list(self.main_preset.blocks),
            num_classes=self.num_classes,
            grid_sizes=self.grid_sizes,
            inference_only=True,
            max_detections=self.max_detections,
        )
        reference = next(self.parameters())
        main.to(device=reference.device, dtype=reference.dtype)
        main.load_state_dict({
            key: value for key, value in self.state_dict().items()
            if not key.startswith("auxiliary_network.")
        })
        return main.eval()


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("yolov10")
