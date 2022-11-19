

we use 2022 MegCup 炼丹大赛 train dataset, the [data](https://studio.brainpp.com/dataset/3736?name=raw%20%E9%99%8D%E5%99%AA%E6%95%B0%E6%8D%AE%E9%9B%86)

my HDC_MSNet_1 network：Parameter=0.077512M.FLOPs=0.7755G

## train dataset:数据集包含 8192 对 256 x 256 的图像，存储为两个文件，分别表示输入、参考输出。
## test dataset: 数据集包含 1024 个 256 x 256 的图像，用同样的格式存储。
## 提交模型要求：模型拥有不超过 100K 个参数
## Training
select 80% train data as train daset
```
python train_MPRNet.py
```


## Evaluation
select 20% train data as valid data

#### Testing on MegCup dataset
- Download test Data from [here](https://studio.brainpp.com/dataset/3736?name=raw%20%E9%99%8D%E5%99%AA%E6%95%B0%E6%8D%AE%E9%9B%86)
```
python test_HDC_MPRNet.py --save_images
```
we will save prediction file is .bin

As shown in the ![figure](submit/21_Megcup.png), the results we finally achieved in the preliminary competition wasted a lot of time because it was the first time to come into contact with this field, resulting in the model not being fully trained.


our team![hear](submit/team.png)
