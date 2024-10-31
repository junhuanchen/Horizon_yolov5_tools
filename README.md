# Horizon_yolov5_tools
基于地平线RDK开发板部署yolov5的相关工具库项目。  
如有任何反馈欢迎戳邮箱：zxy_yys_leaf@163.com  

- Author: Leaf
- Date: 2024-10-30
## 2024/10/31重要更新
- 重新撰写了`fast_postprocess`包，大幅度优化计算逻辑，提升`yolov5`后处理速度，处理耗时平均仅需几毫秒。
- 解决了指针未释放和大量野指针问题。
- 优化了切片逻辑，现在后处理计算线程的数量可以自行调控。
- 无需设定算法中输出results的最大行数，因为算法内部将后处理计算与目标数量`obj_num`的获取分离了。

## 使用方法
### 编译后处理库
进入目录`Horizon_yolov5_tools/fast_postprocess`，键入`sudo python3 setup.py build_ext --inplace`进行编译，弹出有关`Numpy API`的警告可忽略，看见绿色字样`Setup has been completed!`后即编译成功。  

在目录`Horizon_yolov5_tools`下的python脚本中，可以通过：
``` python
from fast_postprocess import RDKyolov5postprocess
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
--thread_num <后处理线程数> \
--nice <程序优先级> \
--CPUOC <是否CPU超频>
```

- `--model`: 可以指定需要推理的模型路径，但必须是去除了yolov5的输出解码层导出，拥有3个输出头的onnx模型使用地平线工具链量化的bin文件模型。详情可参考[【模型提速】如何在X3pi使用yolov5模型50ms推理](https://developer.horizon.cc/forumDetail/163807123501918330)，demo中的模型仅适用于J3/X3芯片，J5/X5或其他版本模型需要另外指定，否则无法运行。   
- `--image`: 指定推理的图片  
- `--names`: 储存模型各个类别名称的yaml文件，参考`models/demo.yaml`
- `--sco_thr`: 得分阈值，默认0.4
- `--nms_thr`: NMS阈值，默认0.45
- `--swap_num`: 后处理线程数，默认为8，建议设置与CPU核心数相同
- `--nice`: 程序优先级，范围通常是从-20（最高优先级）到19（最低优先级），经过尝试发现对运算的加速效果一般
- `--CPUOC`: 指定为`True`可以在程序运行时开启CPU超频加速运算，效果不错

## fast_postprocess说明 - 2024/10/31更新 -
C++快速后处理库实现。  
函数通过处理推理输出的3个`boxes`数组的指针，根据设定的线程数进行切片并放到多线程函数中并行遍历筛选、解码。  
算法运用了C++代替python计算、多线程加速和先验算法逻辑进行了全方面加速：  
- 筛选、解码函数中，会先对输入的切片索引做判断，将3个`boxes`的总索引映射到单个`boxes`。  
- 然后先对得分阈值`score_threshold`进行反`sigmoid`激活，对于每个`box`都会利用`sigmoid`的 单调性，先判断`conf`得分是否超过反激活阈值（因为`score = sigmoid(conf) * sidmoid(prob)`而`sidmoid(prob)`的值在`0`到`1`之间），这一逻辑节约了大量的sigmoid计算。  
- 通过`conf`得分筛选后再搜索最大得分的类和其id，用同样的方式对`prob`得分做判断，如果通过，再计算真正的`score`得分并与`score_threshold`比较筛选。  
-  `score`达到阈值后函数才会结合其他部分条件继续做筛选，所有筛选都通过后才会继续激活位置信息并解算为`[xmin ymin xmax ymax cls_id score]`格式box并加入筛选成功数组中。  
- 最后将所有线程的筛选解码结果整合，进行NMS筛选，输出最终处理结果。  

### `RDKyolov5postprocess.hpp`
C++快速后处理函数的主要内容。  

``` c++
#ifndef RDKYOLOV5POSTPROCESS_HPP
#define RDKYOLOV5POSTPROCESS_HPP

class RDKyolov5postprocess{
    private:
        // 常量
        const int stride[3] = {8, 16, 32};
        const int anchors[3][3][2] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
        // 接收参数
        int model_size;     // 模型尺寸
        int classes_number;    // 类别数量
        float score_threshold; // 得分阈值
        float nms_threshold;   // NMS阈值
        int thread_num;        // 线程数
        // 内部计算参数
        float deactivate_score; // 反激活得分阈值
        int boxes_num[3];       // 3个输出头的box数
        int sub_box_num;        // 子线程任务box数
        int end_box_num;        // 尾线程任务box数
        int dboxes_num;         // 解码容器计数器
        float* decode_boxes;    // 解码容器
        int* nms_list;          // NMS标识列表
        int vim_num;            // 目标数量

    public:
        RDKyolov5postprocess(
            int model_size,     // 模型尺寸
            int classes_number,    // 类别数量
            float score_threshold, // 得分阈值
            float nms_threshold,   // NMS阈值
            int thread_num);       // 线程数

        int process(
            float* boxes0, // 3个box指针
            float* boxes1,
            float* boxes2);

        void score_filter(
            float* boxes0,   // 3个box的地址
            float* boxes1, 
            float* boxes2, 
            int begin_index, // 开始过滤的首索引
            int filter_len); // 过滤长度

        void get_results(
            float* results); 

        ~RDKyolov5postprocess();
};

#endif
```

- `RDKyolov5postprocess::RDKyolov5postprocess`: 构造函数  

- `void RDKyolov5postprocess::process`: 后处理函数  

- `int RDKyolov5postprocess::score_filter`: 得分筛选与解码函数，用于多线程计算服务  
  
- `void RDKyolov5postprocess::get_results`: 获取结果  
  
- `RDKyolov5postprocess::~RDKyolov5postprocess`: 析构函数  

### `__postprocess__.pyx`
将`C++ API`包装转换到`Python API`,同时也简化了函数使用。  

``` cython
cdef class yolov5postprocess:
    cdef RDKyolov5postprocess* c_obj

    def __cinit__(
        self, 
        model_size, 
        classes_number, 
        score_threshold = 0.4, 
        nms_threshold = 0.45, 
        thread_num = 4
        ):
        self.c_obj = new RDKyolov5postprocess(model_size, classes_number, score_threshold, nms_threshold, thread_num)

    def __dealloc__(self):
        del self.c_obj

    def process(self, outputs):
        results_lenth = self.c_obj.process(
            <float*>cnp.PyArray_DATA(outputs[0].buffer.reshape(-1)),
            <float*>cnp.PyArray_DATA(outputs[1].buffer.reshape(-1)),
            <float*>cnp.PyArray_DATA(outputs[2].buffer.reshape(-1)))
        results = np.zeros((results_lenth,6), dtype=np.float32)
        self.c_obj.get_results(<float*>cnp.PyArray_DATA(results))
        return results
```

- `yolov5postprocess.__cinit__`: 构造函数。  
  
- `yolov5postprocess.__dealloc__`: 析构函数。  
  
- `yolov5postprocess.process`: 后处理函数。  
  

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
