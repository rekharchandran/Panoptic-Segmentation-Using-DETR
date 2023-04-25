# Panoptic-Segmentation-Using-DETR
Panoptic Segmentation.

### Panoptic architecture overview:

![image](https://user-images.githubusercontent.com/50706192/204527900-93e1ebf5-c886-49b2-ab38-6e97159c9c0f.png)

•	 We first feed the image to the CNN and set aside the activations from intermediate layers - Res5, Res4, Res3, Res2.
• These are then passed to transformer encoder, after the encoder we also set aside the encoder version of the image and then proceed to the decoder.
•	 we endup with object embedding for the foreground objects and for each segment of the background objects
•	 Next, a multi-head attention layer is used that returns the attention scores over the encoded image for each object embedding.
•	 we proceed to upsample and clean these masks, using convolutional network that uses the imtermediate activations from the backbone.
•	 As a result we get high resolution maps where each pixel contains a binary logit of belonging to the mask.

•	FInally the masks are merged by assigning each pixel to the mask with the highest logit using a simple pixel wise argmax
