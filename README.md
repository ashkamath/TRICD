# TRICD: Testing Robust Image Understanding Through Contextual Phrase Detection

This repository contains evaluation code and access to the pubically available validation split for the TRICD dataset. TRICD is designed to probe multimodal text-visual processing systems on the novel task of Contextualized Phrase Detection (CPD) proposed in the accompanying paper.

CPD is a task that requires models to demonstrate proficiency in both visual object classification (determination of whether or not an object is present in an image) and localization (identification of where, if present, an object appears in the visual scene). Essentially this is an extension of traditional computer vision object detection tasks where the label set is known a priori to an open vocabulary setting. This task is distinct from existing benchmarks designed to probe visio-linguistic understanding such as phrase grounding or referring expression comprehension (REC) because these tasks assume objects or scenes described are necessarily present in their accompanying image; it is merely the system's job to identify where or perform the localization task. Because of this, models do not necessarily need to understand objects in context. We demonstrate that these benchmarks overestimate model performance in comparison to this new evaluation set we propose. 

TL;DR the dataset consists of instances of image-text pairs (two images and two captions) with bounding boxes for each phrase present in each of the images. The image pairs are contextually related, but partially contradictory: while images and text pairs have semanitc similarities, the phrases or objects in each sentence are only present in one of the images. Crucially, this means there are confirmed negatives in this dataset, allowing for a true dual classification and localization task. For a model to perform well it must refrain from predicting bounding boxes when an object is not present. For a more detailed discussion of the dataset design please see how our paper.

## Sub-tasks
We propose two variations on the CPD task: a grounding formaulation and a VQA formulation. 

* **CPD Grounding** : Similar to phrase grounding, for CPD grounding, a model must make bounding box predictions for all annotated phrases within an image caption. 
* **CPD VQA** :  This formulation is quite straightforward; a system must simply answer whether or not certain objects or scenes are present in a given image.

## Dataset splits
We use two main image sources:
1. Winoground : This dataset is also composed of image-caption pairs where the two images and accompnaying texts are semantically similar, often containing the exact same words but in a different ordering for each caption.
2. MS COCO test dataset: We manually selected and annotated a subset of image pairs from the test split that would be challenging and contextually confusing for visio-linguistic systems. The two main types of confounding image-text pairs are those with objects appearing in surprising contexts and those where objects appear in surprising relation to one another

Thus we report metrics on 3 dataset splits
  * Winoground 
  * COCO-obj: COCO images selected for surprising objects
  * COCO-rel: COCO image sselected for surpirsing relations
 
For a more in-depth discussion of the dataset design and splits please see Sections 3 and 4 of the paper. 

## Evaluation
We run inference on the TRICD test and val splits on several current SOTA models. We evaluate 6 systems on the CPD grounding subtask (4 grounding models and 2 open vocabulary detection models) and 3 systems on the VQA subtask. 




