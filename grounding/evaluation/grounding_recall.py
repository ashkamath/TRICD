from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import json
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm
import torch

from box_utils import BoxList, BoxFormat

class RecallTracker:
    """Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report

def get_recall(gt_data, pred_data,  topk=[1], iou_thresh=0.5):
    
    recall_tracker = RecallTracker(topk)
    pid2gt = defaultdict(list)
    for ann in gt_data["annotations"]:
        pid2gt[str(ann["phrase_id"])].append(ann["bbox"])
    for img in tqdm(gt_data["images"]):
        if not img["positive"]:
            continue
          
        phrase_type = None
        if img["source"] == "winoground":
            phrase_type = "winoground"
        elif img["source"] == "coco_test2017":
            phrase_type = "coco_obj" if img["coco_type"] == "object" else "coco_rel"
            
        pid2pred = defaultdict(list)
        datum = pred_data[str(img["id"])] if str(img["id"]) in pred_data else pred_data[img["id"]]
        assert len(datum["scores"]) == len(datum["phrase_ids"]) == len(datum["boxes"]) 
        all_preds = []
        for s,pid, b in zip(datum["scores"],datum["phrase_ids"],datum["boxes"]):
            all_preds.append({"s":s, "box":b, "pid": pid})
            
        # Sort by decreasing confidence
        all_preds = sorted(all_preds, reverse=True, key=lambda x:x["s"])
        for p in all_preds:
            pid2pred[str(p["pid"])].append(p["box"])
        
        for pid in img["phrases"]:
            pid = str(pid)
            if len(pid2pred[pid])==0:
                   print(img)
            target_boxes = BoxList(torch.as_tensor(pid2gt[pid]), BoxFormat.XYWH)
            if len(pid2pred[pid])==0:
                ious = torch.zeros(100, len(pid2gt[pid]))
            else:
                cand_boxes = BoxList(torch.as_tensor(pid2pred[pid]), BoxFormat.XYXY)

                ious = BoxList.box_iou(cand_boxes, target_boxes)
            
            #print(pid, ious)
            for k in topk:
                maxi = 0
                if k == -1:
                    maxi = ious.max()
                else:
                    assert k > 0
                    maxi = ious[:k].max()
                if maxi >= iou_thresh:
                    recall_tracker.add_positive(k, "all")
                    recall_tracker.add_positive(k, phrase_type)
                    #for phrase_type in phrase["phrase_type"]:
                    #    recall_tracker.add_positive(k, phrase_type)
                else:
                    recall_tracker.add_negative(k, "all")
                    recall_tracker.add_negative(k, phrase_type)
                    #for phrase_type in phrase["phrase_type"]:
                    #    recall_tracker.add_negative(k, phrase_type)
        

    results = recall_tracker.report()
    table = PrettyTable()
    all_cat = sorted(list(results.values())[0].keys())
    table.field_names = ["Recall@k"] + all_cat

    score = {}
    for k, v in results.items():
        cur_results = [v[cat] for cat in all_cat]
        header = "Upper_bound" if k == -1 else f"Recall@{k}"

        for cat in all_cat:
            score[f"{header}_{cat}"] = v[cat]
        table.add_row([header] + cur_results)

    print(table)
    return results[topk[0]]


def get_group_recall(gt_data, pred_data, topk=[1], iou_thresh=0.5):
    recall_tracker = RecallTracker(topk)
    pid2gt = defaultdict(list)
    for ann in gt_data["annotations"]:
        pid2gt[str(ann["phrase_id"])].append(ann["bbox"])
        
    id2img = {}
    for img in gt_data["images"]:
        cur_id = f"{img['source']}_{img['original_id']}_{int(img['positive'])}"
        assert cur_id not in id2img
        id2img[cur_id] = img

    for img in tqdm(gt_data["images"]):
        if not img["positive"]:
            continue
        
        orig_id = img["original_id"].split("_")
        cur_neg_id = f"{img['source']}_{orig_id[0]}_{1-int(orig_id[1])}_0"
        neg_img = id2img[cur_neg_id]
        phrase_type = None
        if img["source"] == "winoground":
            phrase_type = "winoground"
        elif img["source"] == "coco_test2017":
            phrase_type = "coco_obj" if img["coco_type"] == "object" else "coco_rel"
            
        neg_pid2pos_pid = {}
        for old_pid, old_p in neg_img["phrases"].items():
            matched = False
            for new_pid, new_p in img["phrases"].items():
                if old_p == new_p:
                    neg_pid2pos_pid[int(old_pid)] = int(new_pid)
                    matched = True
                    break
            if not matched:
                assert False, (img, neg_img, old_p)
            

        pid2pred_box = defaultdict(list)
        pid2pred_img = defaultdict(list)
        datum = pred_data[str(img["id"])] if str(img["id"]) in pred_data else pred_data[img["id"]]
        datum["img_id"] = [int(img["id"]) for _ in range(len(datum["scores"]))]
        datum_neg = pred_data[str(neg_img["id"])] if str(neg_img["id"]) in pred_data else pred_data[neg_img["id"]]
        datum_neg["phrase_ids"] = [neg_pid2pos_pid[p] for p in datum_neg["phrase_ids"]]
        
        datum["scores"] += datum_neg["scores"]
        datum["phrase_ids"] += datum_neg["phrase_ids"]
        datum["boxes"] += datum_neg["boxes"]
        datum["img_id"] += [int(neg_img["id"]) for _ in range(len(datum_neg["boxes"]))]
        
        assert len(datum["scores"]) == len(datum["phrase_ids"]) == len(datum["boxes"]) 
        all_preds = []
        for s,pid, b, iid in zip(datum["scores"],datum["phrase_ids"],datum["boxes"], datum["img_id"]):
            all_preds.append({"s":s, "box":b, "pid": pid, "img_id": iid})
            
        # Sort by decreasing confidence
        all_preds = sorted(all_preds, reverse=True, key=lambda x:x["s"])
        for p in all_preds:
            pid2pred_box[str(p["pid"])].append(p["box"])
            pid2pred_img[str(p["pid"])].append(int(p["img_id"]))
        
        #print(img)
        for pid in img["phrases"]:
            pid = str(pid)
            if len(pid2pred_box[pid])==0:
                   print(img)
            target_boxes = BoxList(torch.as_tensor(pid2gt[pid]), BoxFormat.XYWH)
            if len(pid2pred_box[pid])==0:
                ious = torch.zeros(100, len(pid2gt[pid]))
            else:
                cand_boxes = BoxList(torch.as_tensor(pid2pred_box[pid]), BoxFormat.XYXY)

                ious = BoxList.box_iou(cand_boxes, target_boxes)
                
                cand_pred_ids = torch.as_tensor(pid2pred_img[pid])
                assert len(cand_boxes) == len(cand_pred_ids)
                
                wrong_pid = cand_pred_ids != int(img["id"])
                wrong_pid = wrong_pid[:, None].repeat(1, len(target_boxes))
                ious.masked_fill_(wrong_pid, -1)
                #print(wrong_pid.sum().item(),ious[wrong_pid])
            
            #print(pid, ious)
            for k in topk:
                maxi = 0
                if k == -1:
                    maxi = ious.max()
                else:
                    assert k > 0
                    maxi = ious[:k].max()
                if maxi >= iou_thresh:
                    recall_tracker.add_positive(k, "all")
                    recall_tracker.add_positive(k, phrase_type)
                    #for phrase_type in phrase["phrase_type"]:
                    #    recall_tracker.add_positive(k, phrase_type)
                else:
                    recall_tracker.add_negative(k, "all")
                    recall_tracker.add_negative(k, phrase_type)
                    #for phrase_type in phrase["phrase_type"]:
                    #    recall_tracker.add_negative(k, phrase_type)
        

    results = recall_tracker.report()
    table = PrettyTable()
    all_cat = sorted(list(results.values())[0].keys())
    table.field_names = ["Recall@k"] + all_cat

    score = {}
    for k, v in results.items():
        cur_results = [v[cat] for cat in all_cat]
        header = "Upper_bound" if k == -1 else f"Recall@{k}"

        for cat in all_cat:
            score[f"{header}_{cat}"] = v[cat]
        table.add_row([header] + cur_results)
        
    print(table)
    return results[topk[0]]


