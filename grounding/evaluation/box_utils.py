from __future__ import annotations
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from torchvision.ops.boxes import box_area

from collections import defaultdict
from enum import Enum, auto
from functools import singledispatch, singledispatchmethod
from typing import Any, Dict, Optional, Tuple
from pandas import DataFrame
import numpy as np

import torch
from torchvision.ops.boxes import box_iou



def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def obj_to_box(obj: Dict[str, Any]):
    """Extract the bounding box of a given object as a list"""
    return [obj["x"], obj["y"], obj["w"], obj["h"]]


def region_to_box(obj: Dict[str, Any]):
    """Extract the bounding box of a given region as a list"""
    return [obj["x"], obj["y"], obj["width"], obj["height"]]


class BoxFormat(Enum):
    XYXY = auto()  # pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    XYXY_Normalized = (
        auto()
    )  # pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y] normalized by image size
    XYWH = auto()  # Coco format [top_left_x, top_left_y, width, height]
    CxCyWH = auto()  # Custom format [center_x, center_y, width, height]
    XYWH_Normalized = auto()  # Coco format [top_left_x, top_left_y, width, height] normalized by image size


class BoxList:
    """Helper class to manage a list of boxes transparently based on the format"""

    def __init__(
        self,
        boxes: Optional[torch.Tensor],
        box_format: BoxFormat,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Constructs a BoxList from a torch tensor of boxes

        Args:
            box_list(torch.Tensor): Tensor of shape [N,4] or [4] containing the boxes.
            box_format(BoxFormat): Indicates the format in which the given boxes are stored
            image_size(Tuple[int, int]) Size w,h of the image. Used for normalization
        """
        self.cur_format = box_format
        self.boxes = boxes.float() if boxes is not None else torch.zeros(0, 4)
        assert self.boxes.shape[-1] == 4
        self.img_size = image_size
        if self.boxes.ndim == 1:
            self.boxes = self.boxes.unsqueeze(0)

        if self.img_size is not None:
            self.convert(BoxFormat.XYXY)
            w, h = self.img_size
            self.boxes[:, 0::2].clamp_(min=0, max=w)
            self.boxes[:, 1::2].clamp_(min=0, max=h)
            self.convert(box_format)

        elif box_format in [BoxFormat.XYXY_Normalized, BoxFormat.XYWH_Normalized]:
            self.convert(BoxFormat.XYXY_Normalized)
            self.boxes[:, 0::2].clamp_(min=0, max=1)
            self.boxes[:, 1::2].clamp_(min=0, max=1)
            self.convert(box_format)

    @classmethod
    def fromlist(
        cls,
        box_list: list,
        box_format: BoxFormat,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """Constructs a BoxList from a list of boxes

        Args:
            box_list(list): List of all the boxes
            box_format(BoxFormat): Indicates the format in which the given boxes are stored
        """
        return cls(torch.as_tensor(box_list).float(), box_format, image_size)

    @classmethod
    def frompandas(cls, df: DataFrame, box_format: BoxFormat):
        """Constructs a BoxList from a Pandas dataframe
        Args:
            df: Dataframe containing at least the following columns:
               "height", "width" Size of the image (assumed to be constant on all rows)
               "bbox" list of 4 coordinates corresponding to the box
            box_format(BoxFormat): Indicates the format in which the given boxes are stored
        """
        image_size = None
        if "width" in df.columns and "height" in df.columns:
            image_size = (fast_df_get_column(df, "width")[0], fast_df_get_column(df, "height")[0])

        return cls(
            torch.from_numpy(np.stack(fast_df_get_column(df, "bbox"))),
            box_format,
            image_size,
        )

    def __len__(self):
        return len(self.boxes)

    def __str__(self):
        return f"BoxList in format {self.cur_format}: {self.boxes}"

    def __repr__(self):
        return f"BoxList in format {self.cur_format}: {self.boxes}"

    @staticmethod
    def box_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
        """Box format conversion utility

        Args:
            boxes(torch.Tensor): boxes in pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y].

        Returns:
            boxes in Coco format [top_left_x, top_left_y, width, height]
        """
        assert boxes.shape[-1] == 4
        converted = boxes.clone()
        converted[..., 2:] -= converted[..., :2]
        assert (converted[..., 2:] >= 0).all().item(), "Found empty boxes"
        return converted

    @staticmethod
    def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Box format conversion utility

        Args:
            boxes(torch.Tensor): boxes in Coco format [top_left_x, top_left_y, width, height].

        Returns:
            boxes in pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """
        assert boxes.shape[-1] == 4
        converted = boxes.clone()
        converted[..., 2:] += converted[..., :2]
        return converted

    @staticmethod
    def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """Box format conversion utility

        Args:
            boxes(torch.Tensor): boxes in pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y].

        Returns:
            boxes in format [center_x, center_y, width, height]
        """
        assert boxes.shape[-1] == 4
        x0, y0, x1, y1 = boxes.clone().unbind(-1)
        converted = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)
        assert (converted[..., 2:] >= 0).all().item(), "Found empty boxes"
        return converted

    @staticmethod
    def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Box format conversion utility

        Args:
            boxes(torch.Tensor): boxes in format [center_x, center_y, width, height].

        Returns:
            boxes in pascal format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """
        assert boxes.shape[-1] == 4
        x_c, y_c, w, h = boxes.clone().unbind(-1)
        return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1)

    def is_normalized(self) -> bool:
        return self.cur_format in [BoxFormat.XYWH_Normalized, BoxFormat.XYXY_Normalized]

    def convert(self, target_format: BoxFormat) -> BoxList:
        if self.cur_format == target_format:
            # Nothing to do
            return self

        need_denorm = target_format in [
            BoxFormat.XYWH_Normalized,
            BoxFormat.XYXY_Normalized,
        ]
        if self.cur_format in [BoxFormat.XYWH_Normalized, BoxFormat.XYXY_Normalized]:
            if target_format not in [
                BoxFormat.XYWH_Normalized,
                BoxFormat.XYXY_Normalized,
            ]:
                assert self.img_size is not None, "Need image size to handle normalized conversions"
                self.boxes = self.boxes * torch.as_tensor(
                    [
                        self.img_size[0],
                        self.img_size[1],
                        self.img_size[0],
                        self.img_size[1],
                    ]
                ).unsqueeze(0).to(self.boxes)
            else:
                need_denorm = False
            # proceed as if we were denormalized

            if self.cur_format == BoxFormat.XYWH_Normalized:
                self.cur_format = BoxFormat.XYWH
            elif self.cur_format == BoxFormat.XYXY_Normalized:
                self.cur_format = BoxFormat.XYXY

        if self.cur_format == target_format:
            # Nothing to do
            return self

        if target_format == BoxFormat.XYXY:
            if self.cur_format == BoxFormat.XYWH:
                self.boxes = BoxList.box_xywh_to_xyxy(self.boxes)
            elif self.cur_format == BoxFormat.CxCyWH:
                self.boxes = BoxList.box_cxcywh_to_xyxy(self.boxes)
            else:
                raise RuntimeError(f"Unsupported conversion to XYXY from format {self.cur_format}")
            self.cur_format = target_format
            return self

        # We use XYXY as a pivot format, because we have all the conversions from and to this format
        self.convert(BoxFormat.XYXY)
        assert self.cur_format == BoxFormat.XYXY
        if target_format == BoxFormat.XYWH:
            self.boxes = BoxList.box_xyxy_to_xywh(self.boxes)
        elif target_format == BoxFormat.XYWH_Normalized:
            self.boxes = BoxList.box_xyxy_to_xywh(self.boxes)

            if need_denorm:
                assert self.img_size is not None, "Need image size to handle normalized conversions"
                self.boxes = self.boxes / torch.as_tensor(
                    [
                        self.img_size[0],
                        self.img_size[1],
                        self.img_size[0],
                        self.img_size[1],
                    ]
                ).unsqueeze(0).to(self.boxes)
        elif target_format == BoxFormat.XYXY_Normalized:
            if need_denorm:
                assert self.img_size is not None, "Need image size to handle normalized conversions"
                self.boxes = self.boxes / torch.as_tensor(
                    [
                        self.img_size[0],
                        self.img_size[1],
                        self.img_size[0],
                        self.img_size[1],
                    ]
                ).unsqueeze(0).to(self.boxes)
        elif target_format == BoxFormat.CxCyWH:
            self.boxes = BoxList.box_xyxy_to_cxcywh(self.boxes)
        else:
            raise RuntimeError(f"Unsupported conversion from XYXY to format {target_format}")
        self.cur_format = target_format
        return self

    @staticmethod
    def box_iou(boxes1: BoxList, boxes2: BoxList) -> torch.Tensor:
        """Helper to compute pair-wise Intersection over Union (IoU)

        Args:
            boxes1(BoxList): first set of N boxes
            boxes2(BoxList): first set of M boxes
        Returns:
            iou(torch.Tensor) a [N,M] matrix such that iou[i,j] is the iou between boxes1[i] and boxes2[j]
        """
        if boxes1.is_normalized() and boxes2.is_normalized():
            boxes1.convert(BoxFormat.XYXY_Normalized)
            boxes2.convert(BoxFormat.XYXY_Normalized)
        else:
            boxes1.convert(BoxFormat.XYXY)
            boxes2.convert(BoxFormat.XYXY)
        return box_iou(boxes1.boxes, boxes2.boxes)

    def get_equivalent_boxes(self, iou_threshold=0.95):
        """Find clusters of highly overlapping boxes
        Parameters:
            - iou_threshold: threshold at which we consider two boxes to be the same

        Returns:
            a dict where the keys are an arbitrary id, and the values are the equivalence lists
            (ids of boxes that are equivalent)
        """
        if len(self.boxes) == 0:
            return {0: []}
        uf = UnionFind(len(self.boxes))

        iou = BoxList.box_iou(self, self)
        ind_i, ind_j = torch.where(iou >= iou_threshold)
        for i, j in zip(ind_i.tolist(), ind_j.tolist()):
            uf.unite(i, j)
        compo = defaultdict(list)
        for i in range(len(self.boxes)):
            compo[uf.find(i)].append(i)
        return compo

    def get_smallest_enclosing_box(self) -> BoxList:
        """Return the smallest box that contains all our boxes"""
        self.convert(BoxFormat.XYXY)
        hull = [
            min(self.boxes[:, 0]).item(),
            min(self.boxes[:, 1]).item(),
            max(self.boxes[:, 2]).item(),
            max(self.boxes[:, 3]).item(),
        ]
        return BoxList.fromlist(hull, BoxFormat.XYXY, image_size=self.img_size)

    def get(self, target_format: BoxFormat) -> torch.Tensor:
        """Access the box in a given format"""
        self.convert(target_format)
        return self.boxes

    @singledispatchmethod
    def append(self, boxes):
        raise NotImplementedError("Append not implemented for this type")

    @append.register
    def _(self, other: torch.Tensor, box_format: BoxFormat):
        return self.append(BoxList(other, box_format=box_format))

    @append.register
    def _(self, other: list, box_format: BoxFormat):
        return self.append(BoxList.fromlist(other, box_format=box_format))


@BoxList.append.register
def _(self, other: BoxList):
    other.convert(self.cur_format)
    self.boxes = torch.cat([self.boxes, other.boxes], dim=0)
    return self
