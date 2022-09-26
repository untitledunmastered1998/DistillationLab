# DistillationLab
untitledunmastered1998/DistillationLab 

## Requirements

experiment eviroument
- python3.8.12
- pytorch1.10.1


## 1. dataset  

|datasets |#train samples|#test samples|#classes|resolution|  
|:-------:|:-------:|:-------:|:-------:|:-------:|  
|CIFAR100|50000|10000|100|low|  
|MNIST|60000|10000|10|low|  
|vggface2|2763078|548208|9131|low|  
|ImageNet|1281167|50000|1000|high|
|ImageNet_subset|12610|5000|100|high|  
|(loader还没写)ImageNet32| | | |low|  
|Tiny-ImageNet|100000|10000|200|low|  
|Cars|8144|8041|196|high|  
|flowers102|2040|6149|102|high|  
|stanford_dogs|12601|8519|120|high|  
|aircrafts|6667|3333|100|high|  

## 2. models
Available teacher and student networks including:  
'resnet32', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',  
'mobilenet_v2',  
'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',  
'squeezenet1_0', 'squeezenet1_1'  

|networks|parameters|  
|:-------:|:-------:|  
|resnet32||   
|ResNet18||   
|ResNet34||   
|ResNet50||   
|ResNet101||  
|ResNet152||  
|mobilenet_v2||  
|shufflenet_v2_x0_5||  
|shufflenet_v2_x1_0||  
|shufflenet_v2_x1_5||  
|shufflenet_v2_x2_0||  
|squeezenet1_0||  
|squeezenet1_1||  

## 3. distillation methods  
① knowledge distillation [Distilling the Knowledge in a Neural Network] (https://arxiv.org/abs/1503.02531)  
② L2  
③ FitNets [FitNets: Hints for Thin Deep Nets] (https://arxiv.org/abs/1412.6550)  
④ PKT [Learning Deep Representations with Probabilistic Knowledge Transfer] ECCV2018 (https://arxiv.org/abs/1803.10837)  
⑤ RKD [Relational Knowledge Distillation] CVPR 2019(https://arxiv.org/abs/1904.05068)  

## 4. setup  
Baseline performance follows standard image classification training procedures.  


## 5. training skills

|tricks|performance|  
|:-------:|:-------:|  
|baseline||   
|+xavier init / kaiming init||   
|+pretrained weights||   
|+no bias decay||   
|+label smoothing||  
|+random erasing||  
|+linear scaling learning rate||  
|+cutout||  
|+dropout||  
|+cosine learning rate decay||  
|+warm up stage||  
|+mixup||  
|+Zero γ||
|data augmentation||
|Learning Rate Schedule||


## 6.experiments

① experiments on CIFAR100

|teacher|ResNet18||ResNet34||ResNet50||ResNet101||ResNet152|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|  
|student|mobilenet_v2|||||  
|teacher baseline|||||  
|student baseline|||||  
|KD|||||  
|FitNets|||||    
|RKD|||||   
|PKT|||||  
|L2|||||  
|AT|||||  
|overhaul|||||    
