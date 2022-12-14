# DistillationLab
untitledunmastered1998/DistillationLab 

## Requirements

experiment environment
- python3.8.12
- pytorch1.10.1


## 1. datasets  

|dataset |#train samples|#test samples|#classes|resolution|  
|:-------:|:-------:|:-------:|:-------:|:-------:|  
|CIFAR100|50000|10000|100|low|  
|MNIST|60000|10000|10|low|  
|vggface2|2763078|548208|9131|low|  
|ImageNet|1281167|50000|1000|high|
|ImageNet_subset|12610|5000|100|high|  
|ImageNet32|1281167|50000|1000|low|
|ImageNet32_reduced|384631|15000|300|low|
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

### experiments on CIFAR100

#### same student, different teacher networks

|teacher|ResNet18|ResNet34|ResNet50|ResNet101|ResNet152|  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|   
|student|mobilenet_v2|||||  
|t_baseline|||||  
|s_baseline|||||  
|KD|||||  
|FitNets|||||  
|RKD|||||   
|PKT|||||   
|L2|||||   
|AT|||||  
|overhaul|||||  

#### different student networks

|teacher|ResNet18|ResNet34|ResNet50|ResNet101|ResNet152|  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|   
|student|mobilenet_v2|shufflenet_v1|squeezenet_v0|shufflenet_v2|WRN-16-2|  
|t_baseline|||||  
|s_baseline|||||  
|KD|||||  
|FitNets|||||  
|RKD|||||   
|PKT|||||   
|L2|||||   
|AT|||||  
|overhaul|||||  

#### same architecture

|teacher|ResNet18|ResNet34|ResNet50|ResNet101|ResNet152|  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|   
|student|resnet8×4|resnet32|resnet18|resnet34|resnet50|  
|t_baseline|||||  
|s_baseline|||||  
|KD|||||  
|FitNets|||||  
|RKD|||||   
|PKT|||||   
|L2|||||   
|AT|||||  
|overhaul|||||


#### cross-model transfer

|datasets|student|KD|AT|L2|FitNets|CRD|RKD|PKT|teacher|  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|   
|CIFAR100→STL-10||||||||||  
|CIFAR100→Tiny-ImageNet||||||||||  





