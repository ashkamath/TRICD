# TRICD: Testing Robust Image Understanding Through Contextual Phrase Detection

This repository contains evaluation code and access to the pubically available validation split for the TRICD dataset. TRICD is designed to probe multimodal (vision & language) systems on the novel task of Contextualized Phrase Detection (CPD) proposed in our accompanying paper.

CPD is a task that requires models to demonstrate proficiency in both visual object classification (determination of whether or not an object is present in an image) and localization (identification of where, if present, an object appears in the visual scene). Essentially this is an extension of the traditional computer vision task of object detection where the label set is known a priori, to an open vocabulary setting where the model must also take into account the full context of the sentence. This task is distinct from existing benchmarks designed to probe vision and language understanding such as phrase grounding or referring expression comprehension (REC) because these tasks assume objects or scenes described are necessarily present in their accompanying image; it is merely the system's job to identify where (localization). Because of this, models do not necessarily need to understand objects in context. We demonstrate that these benchmarks overestimate model performance in comparison to our newly proposed evaluation benchmark (TRICD). 

TL;DR the dataset consists of instances of image-text pairs (two images and two captions) with bounding boxes for each phrase present in each of the images. The image pairs are contextually related, but partially contradictory: while images and text pairs have semanitc similarities, the phrases or objects in each sentence are only present in one of the images. Crucially, this means there are confirmed negatives in this dataset, allowing for a true dual classification and localization task. For a model to perform well it must refrain from predicting bounding boxes when an object is not present. For a more detailed discussion of the dataset design please see our paper.


## Dataset splits
We use two main image sources:
1. Winoground : This dataset is also composed of image-caption pairs where the two images and accompanying texts are semantically similar, often containing the exact same words but in a different ordering for each caption.
2. MS COCO test set: We manually selected and annotated a subset of image pairs from the test split that would be challenging and contextually confusing for vision and language models. The two main types of confounding image-text pairs are those with objects appearing in surprising contexts and those where objects appear in surprising relation to one another.

Thus we report metrics on 3 dataset splits
  * **Winoground** : all Winoground images
  * **COCO objects**: COCO images selected for surprising objects
  * **COCO relations**: COCO image sselected for surpirsing relations
 
For a more in-depth discussion of the dataset design and splits please see Sections 3 and 4 of the paper. 

## Sub-tasks
We support evaluation on two sub-tasks of the CPD task, allowing models with different capabilities to measure performance on our challenge:

* **CPD Grounding** : Similar to phrase grounding, for CPD grounding, a model must make bounding box predictions for all annotated phrases within an image caption, where only the positive pairs are evaluated. 
* **CPD VQA** :  This formulation is quite straightforward; a system must simply answer whether or not certain objects or scenes are present in a given image.


## Evaluation
We run inference on the TRICD test and val splits on several current SOTA models. We evaluate 6 systems on the CPD grounding subtask (4 grounding models and 2 open vocabulary detection models) and 3 systems on the VQA subtask. 

### Conextual Phrase Detection (CPD) Performance (Average Precision)

| Model| Winoground | COCO Objects | COCO Relations | All |
|----------|---------|---------|-----------|----------|
| MDETR| 10.1 |       3.9 |     20.4    |     10.7|
| GLIP-T| 14.7 |       22.5 |     25.1.2 |     16.8| 
| GLIP-L| 18.1 | 26.9 | 28.6 |     20.1 | 
| FIBER|  **19.1** | 25.3 | **31.6** | **21.5**| 
| OWL-VIT| 6.3 | 13.7 | 16.3 |     7.9  | 
| DETIC| 8.7 |  **27.0** | 19.7 |   11.6 | 


### CPD-Grounding Sub-Task Performance (Average Precision)

| Model| Winoground | COCO Objects | COCO Relations | All |
|----------|---------|---------|-----------|----------|
| MDETR| 75.8 |       45.0 |           80.0 |     72.0 |
| GLIP-T| 70.6 |         62.7 |           82.2 |     71.7 | 
| GLIP-L| **76.2** | 71.7 | **86.0** |     **77.5** | 
| FIBER|  74.8 | 68.5 | 85.6 | 76.0| 
| OWL-VIT| 62.3 | 72.0 | 78.2 |     66.9  | 
| DETIC| 51.9 |  70.6 | 67.7 |   57.9 | 


### CPD-VQA Sub-Task Performance (F1)
| Model| Winoground | COCO Objects | COCO Relations | All |
|----------|---------|---------|-----------|----------|
| OFA| 54.3 |     71.7 |      67.7 | 62.0  |
| FIBER| **58.5** |  **75.4** |  **74.7** |  **66.7** | 
| Flamingo3B| 51.7 |     75.3 |       74.2 | 63.3| 
| Flamingo80B |  48.2 | 56.4 |  52.3 |  52.1| 




