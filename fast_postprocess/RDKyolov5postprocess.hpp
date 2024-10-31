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