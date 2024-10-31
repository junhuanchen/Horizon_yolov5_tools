# distutils: language = c++
# cython: language_level=3
cimport numpy as cnp
import numpy as np

# 导入类与成员
cdef extern from "RDKyolov5postprocess.hpp":
    cdef cppclass RDKyolov5postprocess:
        RDKyolov5postprocess(int, int, float, float, int) 
        int process(float*, float*, float*)     
        void get_results(float*)

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