## Humber College Logo Detection

This is a project to detect the Humber College logo in images. The project is done using YOLOv8, an object detection deep learning algorithm. It was done as part of the course "ITE 5201 Introduction to Data Analytics" at Humber College.

**Methodology**

1. Search on Google Images and Bing Search with different keywords such as:

- Humber,
- Humber College,
- Humber College Advertisements,
- Humber College Events,
- Humber College Logo, and
- Humber College Logo (filter: Large size).

2. Download all the images from the searching results with the aid of certain tools.
3. Manually clean the data (by removing unlabellable images).
4. Define Classes "humber_large", "humber_medium", "humber_small":

- **humber_large**: Logo occupies a significant portion of the image, making it the primary focus. Logo should take up more than 50% of the image's area.
- **humber_medium**: The logo is clearly visible and distinguishable, but it is not the dominating feature of the image. This could be when the logoc takes up 10% to 50% of the image area.
- **humber_small**: The logo is present, but it's relatively small in comparison to other elements in the image, possibly requiring more effort to identify. This could be when the logo takes up less than 10% of the image's area.

5. Label the images using YoloLabel program. There are also images that have negative classes (no classes present in the images).
6. Split the images into training, validation and test sets with an 80:10:10 (by `train_test_split` from `sklearn`).

The above has already been done before running the notebook.

**About the dataset**

The dataset is created by manually labelling images of the Humber College logo. There are 1103 images in the dataset. The dataset is split into 3 classes and 1 negative class (no class). The number of images in each class is as follows:

The number of images in each class is as follows:

- Class 0 (humber_large): 58 occurrences,
- Class 1 (humber_medium): 243 occurrences,
- Class 2 (humber_small): 719 occurrences,
- Negative images (without any class labeled): 144.

Note that one image can have multiple occurrences of one or more classes. For example, an image can have both humber_large and humber_medium classes.

**Training**

(Code is run on both Google Colab and local machine. Please refer to the notebooks for more details.)

**Results**

The performance of the model is evaluated using `fitness()` function as follows (from official documentation):

```
def fitness(self):
    """Model fitness as a weighted combination of metrics."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (np.array(self.mean_results()) * w).sum()
```

The model was trained for 300 epochs. The best model (best.pt) was saved at epoch 221. The fitness score of the best model is 0.9537 (with mAP50: **0.98459** and mAP50-95: **0.95031**).

**Inference**

The model was able to detect the Humber College logo in the following image:

![Humber College Logo](https://humber.ca/assets/images/openhouse/new/northcampustour2.jpg)

**Limitation**

Given that the dataset is imbalanced, class weights should have been considered during training. However, the model was trained without considering the class weights. Nonetheless, the model is already performing well given the results of validation.
