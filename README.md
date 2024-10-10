# Horizon_yolov5_tools
基于地平线RDK X3开发板部署yolov5的相关工具库项目。  
如有任何反馈欢迎戳邮箱：zxy_yys_leaf@163.com  
后续会更新RDK X5版本   
- Author: Leaf
- Date: 2024-01-26

## 使用方法
### 编译后处理库
进入目录`Horizon_yolov5_tools/fast_postprocess`，键入`sudo python3 setup.py build_ext --inplace`进行编译，弹出有关`Numpy API`的警告可忽略，看见绿色字样`Setup has been completed!`后即编译成功。  

在目录`Horizon_yolov5_tools`下的python脚本中，可以通过：
``` python
import fast_postprocess.postprocess as pprcoc
```  
或  
``` python
from fast_postprocess import postprocess as pprcoc
```
引入后处理库。  

### yolov5示例代码`yolov5_demo.py`  
在目录`Horizon_yolov5_tools`下键入:
```
sudo python3 yolov5_demo.py
```
会默认使用官方模型`models/yolov5s_672x672_nv12.bin`和官方demo中的图片`images/kite.jpg`进行推理，程序会依次：打印模型信息、验证模型、模型推理、推理结果后处理、显示推理输出信息、输出耗时数据、绘制图形框并保存为`results/kite.jpg`。  

也通过便捷指定参数来运行程序
```
sudo python3 yolov5_demo.py \
--model <bin模型路径> \
--image <推理图片路径> \
--names <names列表yaml文件路径> \
--sco_thr <得分阈值> \
--nms_thr <NMS阈值> \
--swap_num <中间数据长度> \
--max_num <输出结果的最大值> \
--nice <程序优先级> \
--CPUOC <是否CPU超频>
```  
  
- `--model`: 可以指定需要推理的模型路径，但必须是去除了yolov5的输出解码层导出，拥有3个输出头的onnx模型使用地平线工具链量化的bin文件模型。详情可参考[【模型提速】如何在X3pi使用yolov5模型50ms推理](https://developer.horizon.cc/forumDetail/163807123501918330)   
- `--image`: 指定推理的图片  
- `--names`: 储存模型各个类别名称的yaml文件，参考`models/demo.yaml`
- `--sco_thr`: 得分阈值，默认0.4
- `--nms_thr`: NMS阈值，默认0.45
- `--swap_num`: 中间数组的长度，在实现后处理的C++函数中，将模型推理得到的3个boxes分别切片成了4*4,2*2和1*1个子boxes，并且通过21个线程对象并行解码，而解码的结果需要21个长度为`swap_num*6`的数组存取，理论上令`swap_num`为`(model_size/32)^2 * 3`便能保证不会发生指针越界，但实际又不可能达到这个数量，因此此处将这一参数引出可供指定，默认为64，当程序输出`[ERROR]postprocess.cpp: The middle data(size of 84) length out of index!!!`时候说明对应尺寸的boxes发生了越级，需要增大swap_num，但该值过大会导致程序内分配过多指针，可能会降低程序运行速度或增大内存负担
- `--max_num`：输出结果数量的最大值
- `--nice`: 程序优先级，范围通常是从-20（最高优先级）到19（最低优先级），经过尝试发现对运算的加速效果一般
- `--CPUOC`: 指定为`True`可以在程序运行时开启CPU超频加速运算，效果不错

## fast_postprocess说明
C++快速后处理库实现。  
函数通过处理推理输出的3个`boxes`数组的指针，分别切片为`4*4`,`2*2`和`1*1`的`split_boxes`，将21个`split_boxes`送入21个线程中进行并行解码。  
解码函数中，对于每个`box`都会搜索最大得分类和其id，先激活计算得分，通过阈值筛选，如果达到阈值才会继续激活位置信息，否则直接返回，大大降低了耗时。  
然后将21个`boxes`的解码结果整合，进行NMS筛选，输出最终处理结果`best_boxes`和其长度。  
最后通过`Cython`将`C++ API`包装为`Python API`。

### `postprocess.cpp`
C++快速后处理函数的主要内容。  

``` C
int fast_postprocess(
    float *output0,
    float *output1,
    float *output2,
    int model_size,
    int class_number,
    float score_threshold,
    float nms_threshold,
    int middle_data_long,
    float *best_box,
    int box_number);
```  
  
- `float *output0`: 推理的结果输出数组0地址。  

- `float *output1`: 推理的结果输出数组1地址。  

- `float *output2`: 推理的结果输出数组2地址。  
  
- `int model_size`: 模型的输入宽度或长度（长宽必须相同）。  
  
- `int class_number`: 模型类别数量。  
  
- `float score_threshold`: 得分筛选阈值。  
  
- `float float nms_threshold`: 非极大值抑制阈值。  
  
- `int middle_data_long`: 中间数据box的内存分配长度。  

- `float *best_box`: 后处理的结果存放数组，长度为`box_number*6`。  
  
- `int box_number`: 后处理的结果存放数组在外部定义的box长度长度，这里是为了防止指针越界，输出长度必然小于等于该值。  
  
- `return`: 函数返回后处理结果的box数量

### `__postprocess__.pyx`
将`C++ API`包装转换到`Python API`,同时也简化了函数使用。  

``` Python
def postprocess(
    outputs,
    int model_size,
    int class_number,
    origin_size,
    float score_threshold=0.4, 
    float nms_threshold=0.45,
    int swap_num = 64,
    int max_num = 32):
```  

- `outputs`: 直接传入地平线工具链`pyeasy_dnn`库中对象`models[]`的`forward`方法输出对象即可。  
  
- `int model_size`: 模型的输入宽度或长度（长宽必须相同）。  
  
- `int class_number`: 模型类别数量。  
  
- `origin_size`: 原始图片大小`(origin_h, origin_w)`，需要传入元组整数类型数据。  
  
- `float score_threshold`: 得分筛选阈值。  
  
- `float float nms_threshold`: 非极大值抑制阈值。 

- `int swap_num = 64`: 中间数据box的内存分配长度，将被直接传入`(postprocess.cpp:fast_postprocess)middle_data_long`

- `int max_num`: 后处理输出的结果数量上限。

### `setup.py`
用于编译产生Cython的`.so`共享库。  
  
开发板端在目录`fast_postprocess`下键入`sudo python3 setup.py build_ext --inplace`进行编译，弹出有关`Numpy API`的警告可忽略，看见绿色字样`Setup has been completed!`后即编译成功。

## `bpu_yolov5_tools.py`函数说明
一些yolov5板端开发工具函数。
- `model_val(models):`: 验证模型格式是否正确
- `models_info_show(models):`: 打印模型信息
- `bgr2nv12(image):`: cv2图片转nv12数据
- `result_show(data,names):`: 打印显示模型信息
- `get_color(id, class_number):`: 获取图形框颜色
- `result_draw(_img, data, names):`: 绘制图形框
  
### `model_val(models):`
该函数可验证模型是否正确，直接传入地平线工具链`pyeasy_dnn`库中的`models`对象即可进行验证，返回一个布尔值。  
``` python
models = dnn.load(opt.model)
if tools.model_val(models) == 0:
    print('模型验证通过！')
else:
    raise ValueError('模型格式错误！')
```  


### `models_info_show(models):`
该函数用于打印模型信息，同上直接传入模型对象即可，会打印类似如下信息：
```
==================================================
INPUT
--------------------------------------------------
tensor type: NV12_SEPARATE
data type: uint8
layout: NCHW
shape: (1, 3, 672, 672)
==================================================
OUTPUT 0
--------------------------------------------------
tensor type: float32
data type: float32
layout: NHWC
shape: (1, 84, 84, 255)
==================================================
OUTPUT 1
--------------------------------------------------
tensor type: float32
data type: float32
layout: NHWC
shape: (1, 42, 42, 255)
==================================================
OUTPUT 2
--------------------------------------------------
tensor type: float32
data type: float32
layout: NHWC
shape: (1, 21, 21, 255)
==================================================
```

### `bgr2nv12(image):`
传入cv2图片，转换并输出为nv12数据

### `result_show(data,names):`
显示推理结果。
- `data`: numpy数组对象
- `names`：存有模型各类别名称字符串的列表对象。  
  
`data`的结构:`data[n,0]`~`data[n,5]`分别存储`第N个结果信息的xmin,ymin,xmax,ymax,id,score`  
  
会向终端打印类似如下信息：
```
===========================================================================
result
---------------------------------------------------------------------------
ser      xmin     ymin     xmax     ymax     id       name     score    
1        591.52   80.91    673.09   150.87   33       kite     0.86     
2        278.76   236.69   306.49   281.17   33       kite     0.84     
3        576.98   346.23   600.95   370.06   33       kite     0.70     
4        1083.33  395.06   1102.03  422.36   33       kite     0.68     
5        469.29   340.77   486.03   358.94   33       kite     0.59     
6        216.02   696.87   274.15   855.38   0        person   0.85     
7        115.67   615.82   166.85   760.80   0        person   0.78     
8        80.53    510.89   107.69   564.52   0        person   0.69     
9        176.13   541.22   193.60   573.60   0        person   0.64     
10       518.01   508.12   533.78   532.37   0        person   0.61     
11       32.99    510.33   60.70    556.41   0        person   0.52     
12       345.08   487.23   357.90   504.60   0        person   0.51     
13       523.14   510.96   554.48   536.19   0        person   0.46     
14       537.16   514.42   554.66   534.34   0        person   0.45     
15       1205.86  452.89   1215.96  463.19   0        person   0.41     
===========================================================================
```  

### `get_color(id, class_number):`
获取图形框颜色，传入类别id和类个数，返回长度为3的代表bgr值的元组类型。 
  
### `result_draw(_img, data, names):`
绘制图像框，并储存到目录下的`results`目录中。
- `_img`: 传入cv2图片对象
- `data`: 后处理的结果，结构同`result_show(data,names)`的参数`data`  
- `names`: 列表类型，存有模型各类别名称的字符串
- `return`: 返回绘制完成的cv2图片对象
效果可见目录`results`
