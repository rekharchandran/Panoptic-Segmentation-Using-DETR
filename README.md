# Panoptic-Segmentation-Using-DETR
Panoptic Segmentation.
Its is fusion of instance segmentation which aims at predicting a mask for each distinct instant of a foreground object and semantic segmentation which aims at predicting a class label for each pixel in the background, the resulting task requires that each pixel belongs to exactly one segment.

DETR can be naturally extend by adding a mask head on top of the decoder outputs for panoptic segmentation. This head can be used to produce panoptic segmentation by treating stuff and thing classes in a unified way. Through panoptic segmentation the authors aim at understanding whether DETR's object embeddings can be used for the downstream tasks.

### Panoptic architecture overview:

![image](https://user-images.githubusercontent.com/50706192/204527900-93e1ebf5-c886-49b2-ab38-6e97159c9c0f.png)


