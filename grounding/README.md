# CPD-Grounding Sub-Task

This repository provides access to annotations for the validation split of the TRICD dataset as well as code for running evaluation for the following metrics:

* **Recall@K** : recall when considering the top K highest scoring bounding box per phrase 

* **GroupRecall@K**: this can be thought of as a retrieval metric. For each phrase, it considers all predictions made (both for the positive and related negative instance). A result is considered a success if the highest scoring bounding box is associated with the positive prediction and has IOU > 0.5.

* **Average Precision**: standard grounding AP metric 

## Running inference on the val set
To evaluate your models on the TRICD validation set, you will need the COCO2017 validation set images and the [TRICD val annotations](https://github.com/ashkamath/TRICD/blob/main/grounding/annotations/TRICD_grounding_val.json) provided in this repository. 

To download the COCO images:
```
wget http://images.cocodataset.org/zips/val2017.zip 
unzip -q val2017.zip -d <your-image-directory>
```

To access annotations file in python
```
import requests
url = 'https://raw.githubusercontent.com/ashkamath/TRICD/main/grounding/annotations/TRICD_grounding_val.json?token=GHSAT0AAAAAAB55UT7JRLYIBTGYB6HZEILGY7BKNXA'
phrase_det_anns = json.loads(requests.get(url).text)
```



## Expected Results Format
Predictions are expected in the following nested json format where each image_id is a top-level key. Each image should have a list of scores, bounding box predictions
and phrase_id predictions.  The introduction of phrase IDs is the main difference between this and standard grounding results output. The list of phrase IDs identify which positive phrase in a caption a bounding box and score are associated to. 
```
{
  image_id : 
            {
             'scores' : [list of scores],
             'boxes' : [list of bounding box predictions],
             'phrase_ids' : [list of phrase IDs]
             }
}
```

## Usage 
Currently this repository allows functionality for evaluating results on the validation set if the user provides a file of results formatted as above.

We provide basic instructions. First clone this respository locally and install requirements (in a new conda environment if preferred).

```
git clone git@github.com:ashkamath/TRICD.git && cd grounding
pip install -r requirements.txt
```

By default, the evaluation script returns results with metrics Recall@1, Group Recall@1, & AP. The user may specify a subset of these or additional Recall@K
when running the evaluation script. Examples are below:

*Standard run with default metrics*
```
python evaluation/main.py --results_file <path-to-your-model-results>
```

*Just top 1, 5, 10 recall metrics*
```
python evaluation/main.py --results_file <path-to-your-model-results> --metrics recall@k grouprecall@k --topk 1 5 10
```

*Just upper bound for recall*
```
python evaluation/main.py --results_file <path-to-your-model-results> --metrics recall@k --topk -1
```




