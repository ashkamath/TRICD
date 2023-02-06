# TRICD: Testing Robust Image Understanding Through Contextual Phrase Detection

This repository contains evaluation code and access to the pubically available validation split for the TRICD dataset. TRICD is designed to probe multimodal text-visual processing systems on the novel task of Contextualized Phrase Detection (CPD) proposed in the accompanying paper.

CPD is a task that requires models to demonstrate proficiency in both visual object classification (determination of whether or not an object is present in an image) and localization (identification of where, if present, an object appears in the visual scene). Essentially this is an extension of traditional computer vision object detection tasks where the label set is known a priori to an open vocabulary setting. This task is distinct from existing benchmarks designed to probe visio-linguistic understanding such as phrase grounding or referring expression comprehension (REC) because these tasks assume objects or scenes described are necessarily present in their accompanying image; it is merely the system's job to identify where or perform the localization task. Because of this, models do not necessarily need to understand objects in context. We demonstrate that these benchmarks overestimate model performance in comparison to this new evaluation set we propose. 

For a detailed discussion of the dataset design please see how our paper. TL;DR the dataset consists of instances of image-text pairs (two images and two captions) with bounding boxes for each phrase present in each of the images. The image pairs are contextually related, but partially contradictory: while images and text pairs have semanitc similarities, the phrases or objects in each sentence are only present in one of the images. Crucially, this means there are confirmed negatives in this dataset, allowing for a true dual classification and localization task. For a model to perform well it must refrain from predicting bounding boxes when an object is not present. 

## Sub-tasks
We propose two variations on the CPD task: a grounding formaulation and a VQA formulation. 

* **CPD Grounding** : Similar to phrase grounding, for CPD grounding, a model must make bounding box predictions for all annotated phrases within an image caption. 
* **CPD VQA** :  This formulation is quite straightforward; a system must simply answer whether or not certain objects or scenes are present in a given image.

## 


