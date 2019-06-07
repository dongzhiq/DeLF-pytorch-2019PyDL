
# Deep Local Feature (DeLF) 的PyTorch实现
原文"Large-Scale Image Retrieval with Attentive Deep Local Features" https://arxiv.org/pdf/1612.06321.pdf


## 环境要求
+ PyTorch
+ python3
+ CUDA

## 训练DeLF
训练DeLF主要分为两步：微调用ImageNet预训练好的resnet50模型；冻结微调好的网络，只更新用于关键点选择的“注意”网络；训练结束后，训练好的模型保存在`repo/<expr>/keypoint/ckpt`

### (1) 微调模型:
~~~shell
$ cd train/
$ python main.py \
    --stage 'finetune' \
    --optim 'sgd' \
    --gpu_id 6 \
    --expr 'landmark' \
    --ncls 586 \
    --finetune_train_path <path to train data> \
    --finetune_val_path <path to val data> \
~~~

### (2) 训练注意网络:
+ load_from: 要加载的pytorch模型的绝对路径(<model_name>.pth.tar)
+ expr: 保存的实验名
~~~shell
$ cd train/
$ python main.py \
    --stage 'keypoint' \
    --gpu_id 6 \
    --ncls 586 \
    --optim 'sgd' \
    --use_random_gamma_scaling true \
    --expr 'landmark' \
    --load_from <path to model> \
    --keypoint_train_path <path to train data> \
    --keypoint_val_path <path to val data> \
~~~


## 用DeLF进行特征提取
提取DeLF特征也分为两步：训练PCA；提取降维后的DeLF特征  
__注意：必须手动将模型`repo/<expr>/keypoint/ckpt/bestshot.pth.tar`复制（或改名）为`repo/<expr>/keypoint/ckpt/fix.pth.tar`__  
__原作者故意在此设置了手动操作，以免在算完PCA矩阵后模型被误修改__

### (1) 训练PCA
~~~shell
$ cd extract/
$ python extractor.py
    --gpu_id 4 \
    --load_expr 'delf' \
    --mode 'pca' \
    --stage 'inference' \
    --batch_size 1 \
    --input_path <path to train data>, but it is hardcoded.
    --output_path <output path to save pca matrix>, but it is hardcoded.
~~~

### (2) 提取降维后的DeLF特征
~~~shell
$ cd extract/
$ python extractor.py
    --gpu_id 4 \
    --load_expr 'delf' \
    --mode 'delf' \
    --stage 'inference' \
    --batch_size 1 \
    --attn_thres 0.31 \
    --iou_thres 0.92 \
    --top_k 1000 \
    --use_pca True \
    --pca_dims 40 \
    --pca_parameters_path <path to pca matrix file.>, but it is hardcoded.
    --input_path <path to train data>, but it is hardcoded.
    --output_path <output path to save pca matrix>, but it is hardcoded.
~~~


## 可视化
逐行运行visualize.ipynb中的程序，即可将任二图像（test/img1.jpg, test/img2.jpg）的DeLF匹配过程可视化，结果见作者github库（Minchul Shin([@nashory](https://github.com/nashory)) ）。
