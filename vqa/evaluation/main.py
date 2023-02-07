import json
import numpy as np
import os
from sklearn.metrics import f1_score
import argparse
from tabulate import tabulate
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file", help = "File path for ground truth TRICD annotations files", type = str, default = "/../annotations/TRICD_vqa_val.json"
    )
    parser.add_argument(
        "--results_file", help="File location for grounding results to be evaluated", type=str
    )
    return parser


def verify_pred_format(gt_data, pred_data):

    #Verify image_ids in the gt_data match those in the predictions file 
    gt_ids = sorted([i['image_id'] for i in gt_data['questions']])

    #Get predicted image IDs from keys of predictions 
    pred_ids = sorted([int(k) for k in pred_data.keys()])

    assert gt_ids == pred_ids, \
        "Image IDs in ground truth annotations file do not match image IDs in predictions!"
    
    #Make sure results are ints rather than "yes" "no" answers. If not reformat
    ans = sorted(list(set([int(v) for v in pred_data.values()])))
    
    assert ans == [0,1],\
        "Answer format is not correct. Accepted values are 0 (no) and 1 (yes). Please re-format"
        


def VQA_eval(gt_data, pred_data):
    image_ids = [a['image_id'] for a in gt_data['annotations']]
    gt_answers = [a['answer'] for a in gt_data['annotations']]
    pred_answers = [pred_data[str(i)] for i in image_ids]

    results = {}
    results['metric'] = 'F1_score'
    results['all'] = f1_score(gt_answers, pred_answers, average = 'macro')
   
    #Subgroups
    id2source = {a['image_id']: a['source'] if a['source']=='winoground' else a['coco_type'] for a in gt_data['annotations']}
    
    ## COCO_obj
    coco_obj_img_ids = [k for k, v in id2source.items() if v=='object']
    coco_obj_gt_answers = [a['answer'] for a in gt_data['annotations'] if a['image_id'] in(coco_obj_img_ids)]
    coco_obj_pred_answers = [pred_data[str(i)] for i in image_ids]
    results['coco_obj'] = f1_score(coco_obj_gt_answers, coco_obj_pred_answers, average = 'macro')
    
    ## COCO_rel
    coco_rel_img_ids = [k for k, v in id2source.items() if v=='relation']
    coco_rel_gt_answers = [a['answer'] for a in gt_data['annotations'] if a['image_id'] in(coco_rel_img_ids)]
    coco_rel_pred_answers = [pred_data[str(i)] for i in image_ids]
    results['coco_rel'] = f1_score(coco_rel_gt_answers, coco_rel_pred_answers, average = 'macro')

    results_df = pd.DataFrame([results])
    return results_df

def main(args):
    #Load ground truth data from annotations
    with open(os.path.dirname(__file__) + args.gt_file) as f:
        gt_data = json.load(f)

    with open(args.results_file) as f:
        pred_data = json.load(f)
    
    #Verify 
    verify_pred_format(gt_data, pred_data)

    results_df = VQA_eval(gt_data, pred_data)

    print(tabulate(results_df, headers = 'keys', tablefmt = 'psql', showindex = False))




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)