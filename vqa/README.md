# CPD-VQA Sub-Task

This repository provides access to annotations for the validation split of the TRICD dataset as well as code for running evaluation on the CPD visual question answering, or classification, sub-task.


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
url = 'https://raw.githubusercontent.com/ashkamath/TRICD/main/vqa/annotations/TRICD_VQA_val.json'
vqa_anns = json.loads(requests.get(url).text)
```



## Expected Results Format
Predictions are expected as a simple dictionary json where each key is an image ID and each value is an answer. Valid answer values are 0 (no) and 1 (yes). 


## Usage 
Currently this repository allows functionality for evaluating results on the validation set if the user provides a file of results formatted as above.

We provide basic instructions. First clone this respository locally and install requirements (in a new conda environment if preferred).

```
git clone git@github.com:ashkamath/TRICD.git && cd vqa
pip install -r requirements.txt
```

By default, the evaluation script returns F1 for the 3 dataset splits. 

```
python evaluation/main.py --results_file <path-to-your-model-results>
```




