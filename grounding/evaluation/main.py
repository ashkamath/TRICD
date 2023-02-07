import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import contextlib
from sklearn.metrics import f1_score
from itertools import chain 
import pandas as pd
from tabulate import tabulate

from typing import Any, Dict, Optional, Tuple


from box_utils import box_iou, generalized_box_iou, obj_to_box, region_to_box, BoxFormat, BoxList
from grounding_recall import RecallTracker, get_recall, get_group_recall
from grounding_AP import PDEval, Params, get_AP

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file", help = "File path for ground truth TRICD annotations files", type = str, default = "/../annotations/TRICD_grounding_val.json"
    )
    parser.add_argument(
        "--results_file", help="File location for grounding results to be evaluated", type=str
    )
    parser.add_argument(
        "--eval_metrics", nargs="+", help="List of evaluation metrics to generate", default = ['recall@k', 'grouprecall@k', 'ap']
    )
    parser.add_argument(
        "--topk", nargs= "+", help="Top K boxes to evaluate for recall and/or group recall metrics if applicable", default = [1]
    )

    return parser

def verify_pred_format(gt_data, preds_data):
    
    #Verify image_ids in the gt_data match those in the predictions file 
    gt_ids = sorted([i['id'] for i in gt_data['images']])
    
    #Image_ids should be keys in predictions json
    pred_ids = sorted([int(k) for k in preds_data.keys()])
    
    assert gt_ids == pred_ids, \
        "Image IDs in ground truth annotations file do not match image IDs in predictions!"
    
    #Verify phrase_ids in the gt_data match those in the predictions file  
    gt_pids = sorted(list(chain(*[[int(k) for k in i['phrases'].keys()] for i in gt_data['images']])))

    #Get unique predicted phrase_ids
    pred_pids = sorted(list(chain(*[[int(pid) for pid in set(v['phrase_ids'])] for k, v in preds_data.items()])))

    
    assert gt_pids == pred_pids, \
        "Phrase IDs in ground truth annotations do not match phrase IDs in predictions!"
    



def main(args):
    #Load ground truth data from annotations
    with open(os.path.dirname(__file__) + args.gt_file) as f:
        gt_data = json.load(f)

    with open(args.results_file) as f:
        preds_data = json.load(f)
    
    verify_pred_format(gt_data, preds_data)

    topk = [int(k) for k in args.topk]

    results_df = pd.DataFrame()
    if 'recall@k' in args.eval_metrics:
        recall = get_recall(gt_data, preds_data, topk)
        results_df = pd.concat([results_df, recall])

    if 'grouprecall@k' in args.eval_metrics:
        group_recall = get_group_recall(gt_data, preds_data, topk)
        results_df = pd.concat([results_df, group_recall])

    if 'ap' in args.eval_metrics:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                gt_coco = COCO(os.path.dirname(__file__) + args.gt_file)
        
        ap = get_AP(gt_data, gt_coco, preds_data)
        results_df = pd.concat([results_df, ap])
    
    print(tabulate(results_df, headers = 'keys', tablefmt = 'psql', showindex = False))



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

