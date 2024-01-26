# ==================================================
#
#   函数相关API的cython实现
#       * Author: Leaf
#       * Date: 2024-01-26
#
# ==================================================

cimport numpy as cnp
import numpy as np

# 导入C++函数定义
cdef extern from "./postprocess.cpp":
    # int cpp_postprocess(float* ,float* ,float* ,int ,int ,int ,int ,float ,float ,float*)
    int fast_postprocess(float*, float*, float*, int, int, float, float, int, float*, int)

# 适配C++函数 类型转换
def postprocess(
    outputs,
    int model_size,
    int class_number,
    origin_size,
    float score_threshold=0.4, 
    float nms_threshold=0.45,
    int swap_num = 32,
    int max_num = 32):
    """
    model_size: 模型的宽或高（必须一致）
    class_number: 类别数量
    score_threshold: 得分阈值
    nms_threshold: nms阈值
    swap_num: 多线程解码时每个线程的box数，过小会报错，一般32够用
    max_num: 最大输出box数
    """

    results = np.zeros((int(max_num),6), dtype=np.float32)

    cdef int lenth = fast_postprocess(
        <float*>cnp.PyArray_DATA(outputs[0].buffer.reshape(-1)),
        <float*>cnp.PyArray_DATA(outputs[1].buffer.reshape(-1)),
        <float*>cnp.PyArray_DATA(outputs[2].buffer.reshape(-1)),
        <int>model_size,
        <int>class_number,
        <float>score_threshold,
        <float>nms_threshold,
        <int>swap_num,
        <float*>cnp.PyArray_DATA(results),
        <int>max_num)

    results = results[:int(lenth)]
    origin_w, origin_h = origin_size
    results[:,1] *= origin_w/model_size
    results[:,3] *= origin_w/model_size
    results[:,0] *= origin_h/model_size
    results[:,2] *= origin_h/model_size

    return results