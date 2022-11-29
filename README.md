# Panoptic-Segmentation-Using-DETR
Panoptic Segmentation.
Its is fusion of instance segmentation which aims at predicting a mask for each distinct instant of a foreground object and semantic segmentation which aims at predicting a class label for each pixel in the background, the resulting task requires that each pixel belongs to exactly one segment.

DETR can be naturally extend by adding a mask head on top of the decoder outputs for panoptic segmentation. This head can be used to produce panoptic segmentation by treating stuff and thing classes in a unified way. Through panoptic segmentation the authors aim at understanding whether DETR's object embeddings can be used for the downstream tasks.

First task is to train regular DETR to predict boxes around things (foreground) and stuff (background objects) in a uniform manner
Once the detection model is trained we freeze the weights, train a mask head for 25 epochs
