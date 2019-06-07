
# Deep Local Feature (DeLF) ��PyTorchʵ��
ԭ��"Large-Scale Image Retrieval with Attentive Deep Local Features" https://arxiv.org/pdf/1612.06321.pdf


## ����Ҫ��
+ PyTorch
+ python3
+ CUDA

## ѵ��DeLF
ѵ��DeLF��Ҫ��Ϊ������΢����ImageNetԤѵ���õ�resnet50ģ�ͣ�����΢���õ����磬ֻ�������ڹؼ���ѡ��ġ�ע�⡱���磻ѵ��������ѵ���õ�ģ�ͱ�����`repo/<expr>/keypoint/ckpt`

### (1) ΢��ģ��:
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

### (2) ѵ��ע������:
+ load_from: Ҫ���ص�pytorchģ�͵ľ���·��(<model_name>.pth.tar)
+ expr: �����ʵ����
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


## ��DeLF����������ȡ
��ȡDeLF����Ҳ��Ϊ������ѵ��PCA����ȡ��ά���DeLF����  
__ע�⣺�����ֶ���ģ��`repo/<expr>/keypoint/ckpt/bestshot.pth.tar`���ƣ��������Ϊ`repo/<expr>/keypoint/ckpt/fix.pth.tar`__  
__ԭ���߹����ڴ��������ֶ�����������������PCA�����ģ�ͱ����޸�__

### (1) ѵ��PCA
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

### (2) ��ȡ��ά���DeLF����
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


## ���ӻ�
��������visualize.ipynb�еĳ��򣬼��ɽ��ζ�ͼ��test/img1.jpg, test/img2.jpg����DeLFƥ����̿��ӻ������������github�⣨Minchul Shin([@nashory](https://github.com/nashory)) ����
