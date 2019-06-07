
可以通过改变`extract.py`里的超参来训练PCA和获得delf特征文档。

### 超参

~~~

+ MODE:
  
    'pca' or 'delf'.  
  'pca': 提取特征来获得pca矩阵   
  'delf': 提取delf特征并保存到文档   
+ USE_PCA:  
 
	如果想在提取delf特征的时候起用pca降维，令USE_PCA=TRUE。该标识只在MODE='delf'时起作用。

  
+ PCA_DIMS:
  
	如果想在提取delf特征的时候起用pca降维，令USE_PCA=TRUE。该标识只在MODE='delf'时起作用。

+ PCA_PARAMETERS_PATH:
  
	MODE='pca'时: 保存文档pca.h5的路径。 (文档pca.hy5 包含计算后的pca矩阵、pca方差和pca均值)
	
	MODE='delf'时:读取pca matrix 来提取delf特征的路径。

  
+ INPUT_PATH:
  
	要输入进行特征提取的图像的路径。

+ OUTPUT_PATH:
  
	输出delf特征文档的路径。
  该选项只在MODE='delf'时起作用。

  
+ LOAD_FROM:
  
	到希望使用作为特征提取器使用的pytorch模块的路径

~~~


### 如何训练PCA?
在`extract.py`里调整超参，然后run `python extractor.py`

~~~
(example)
MODE = 'pca'
GPU_ID = 0
IOU_THRES = 0.98
ATTN_THRES = 0.17
TOP_K = 1000
USE_PCA = False
PCA_DIMS = 40
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
ARCH = 'resnet50'
EXPR = 'dummy'
TARGET_LAYER = 'layer3'
LOAD_FROM = 'xxx'
PCA_PARAMETERS_PATH = 'xxx'
INPUT_PATH = 'xxx'
OUTPUT_PATH = 'dummy'

python extractor.py
~~~


### 如何提取delf特征？
在`extractor.py`里调整超参，然后run `python extractor.py`

~~~
(example)
MODE = 'delf'
GPU_ID = 0
IOU_THRES = 0.98
ATTN_THRES = 0.17
TOP_K = 1000
USE_PCA = True
PCA_DIMS = 40
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
ARCH = 'resnet50'
EXPR = 'dummy'
TARGET_LAYER = 'layer3'
LOAD_FROM = 'yyy'
PCA_PARAMETERS_PATH = 'yyy'
INPUT_PATH = 'yyy'
OUTPUT_PATH = './output.delf'

python extractor.py
~~~

### [!!!] 注意 [!!!]
+ 尺寸限制：   
  如果宽 * 高 > 1400*1400, 特征会被丢弃以防止GPU内存栈溢出。    
  请保证输入图像的尺寸(w x h)小于1400 * 1400. 
  

