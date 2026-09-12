# MobileNetV3-Large: Model background and proposed pipeline

## Model background
MobileNetV3 is a lightweight CNN architecture developed by Google for efficient computer vision on mobile and resource-constrained devices. It builds on earlier MobileNet architectures and uses techniques such as depthwise separable convolutions, inverted residual blocks, squeeze-and-excitation modules, and hardware-aware architecture optimization to reduce computational requirements while maintaining strong image classification performance. 

Two main versions of the architecture are available: MobileNetV3-Small and MobileNetV3-Large. MobileNetV3-Small is intended for highly resource-constrained environments, while MobileNetV3-Large provides greater model capacity while remaining substantially more computationally efficient than many conventional CNN architectures. MobileNetV3-Large contains approximately 5.5 million parameters, making it relatively compact compared with many standard image classification models. For this specific project, I think it would be more appropriate to use MobileNetV3-Large because the objective is not only to achieve accurate melanoma/skin lesion classification, but also to develop a model that can eventually operate offline on low-power edge hardware such as a Raspberry Pi. The additional capacity of the Large architecture may also be beneficial for identifying subtle visual characteristics of skin lesions, such as differences in colour, texture, shape, and border irregularity, while still maintaining relatively low computational requirements.

MobileNetV3-Large also supports transfer learning, allowing pretrained ImageNet weights to be used as a starting point rather than training the model entirely from scratch. This is particularly relevant to this project because one of the objectives is to achieve reliable performance with limited training data. Its relatively small size also makes it suitable for further optimization through techniques such as INT8 quantization, which can reduce storage and computational requirements for eventual edge deployment.

## Tentative pipeline
The pipeline will use an ImageNet-pretrained MobileNetV3-Large with transfer learning for melanoma classification.

1. Preprocessing and augmentation: images will be resized to 224 × 224 and normalized. Augmentations such as rotation, flipping, brightness changes, blur, and compression will help reduce overfitting and simulate lower-quality images.

2. Transfer learning: a MobileNetV3-Large model pretrained on ImageNet will be used so the model can build on features it has already learned, such as shapes, edges, colours, and textures. Initially, the pretrained layers will be frozen while a new classification layer is trained for melanoma/skin lesion detection. Some of the later layers will then be unfrozen and fine-tuned to better recognize features specific to skin lesions.

3. Evaluation: performance will be measured using metrics such as ROC-AUC, sensitivity, specificity, F1-score, and accuracy. Robustness will also be evaluated using reduced training data and degraded test images.

4. Explainability: Grad-CAM will generate heatmaps highlighting image regions that contributed most strongly to the model's prediction.

5. Optimization and deployement: the trained model will undergo INT8 quantization to reduce model size and computational requirements. The optimized model will then be deployed for offline inference on a Raspberry Pi.

# Pipeline Overview
Skin image -> Preprocessing/augmentation -> MobileNetV3-Large -> Melanoma/skin lesion prediction -> Grad-CAM -> INT8 quantization -> Raspberry Pi deployment
