#include "RDKyolov5postprocess.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <stdexcept>

using namespace std;

// sigmoid
inline float activate(float x){
    return 0.5 * (1 + tanh(0.5 * x));
}

// inverse sigmoid
inline float deactivate(float x){
    return 2 * atanh(2 * x - 1);
}

inline float iou(float box1[4], float box2[4]){
    // box struct: [xmin, ymin, xmax, ymax]
    float left   = max(box1[0], box2[0]);
    float top    = max(box1[1], box2[1]);
    float right  = min(box1[2], box2[2]);
    float bottom = min(box1[3], box2[3]);
    float S1     = (box1[0] - box1[2]) * (box1[1] - box1[3]);
    float S2     = (box2[0] - box2[2]) * (box2[1] - box2[3]);
    if(right-left <= 0 || bottom-top <= 0)return 0.0;
    float interArea = (right-left)*(bottom-top);
    return interArea / (S1 + S2 - interArea);
}

RDKyolov5postprocess::RDKyolov5postprocess(
    int model_size, 
    int classes_number, 
    float score_threshold, 
    float nms_threshold,
    int thread_num
) : model_size(model_size),
    classes_number(classes_number), 
    score_threshold(score_threshold), 
    nms_threshold(nms_threshold),
    thread_num(thread_num),
    deactivate_score(deactivate(score_threshold))
{
    
    // 计算3个boxes中的总box数
    for(int i=0; i<3; i++)boxes_num[i] = 3*model_size*model_size/stride[i]/stride[i];
    // 子线程和尾线程处理的box数
    sub_box_num = (boxes_num[0]+boxes_num[1]+boxes_num[2]) / thread_num;
    end_box_num = (boxes_num[0]+boxes_num[1]+boxes_num[2]) % thread_num;
    // 解码筛选容器 box:
    // [[xmin, ymin, xmax, ymax, class, score], ...,]
    decode_boxes = new float[(boxes_num[0]+boxes_num[1]+boxes_num[2]) * 6]; 
    // 初始化nms选中容器
    nms_list = new int[1];
}

int RDKyolov5postprocess::process(
    float* boxes0,
    float* boxes1,
    float* boxes2
){
    dboxes_num = 0;
    // 创建多线程容器
    vector<thread> score_filter_threads;
    // 将3个输出头切片输入
    for(int i = 0; i < thread_num; i++){
        score_filter_threads.push_back(
            thread(&RDKyolov5postprocess::score_filter, this, boxes0, boxes1, boxes2, i*sub_box_num, sub_box_num));
    }
    // 尾部片段
    if(end_box_num>0)
        score_filter_threads.push_back(
            thread(&RDKyolov5postprocess::score_filter, this, boxes0, boxes1, boxes2, thread_num*sub_box_num, end_box_num));
    // 等待所有线程完成
    for(auto& t:score_filter_threads)t.join();
    // NMS非极大值抑制
    vim_num = 0;
    int max_class_id;
    int max_index;
    float max_score;
    int nms_count = dboxes_num;
    // 重置NMS选中容器并初始化
    delete[] nms_list;
    nms_list = nullptr;
    nms_list = new int[dboxes_num]; 
    for(int i=0; i<dboxes_num; i++)nms_list[i] = 0;
    // nms循环
    while(nms_count > 0){
        // 查询最大score与class_id
        max_class_id = -1;
        max_index = -1;
        max_score = 0.0;
        for(int i=0; i<dboxes_num; i++){
            // 剔除异常矩形框
            if(decode_boxes[i*6+0] >= decode_boxes[i*6+2] || decode_boxes[i*6+1] >= decode_boxes[i*6+3]){
                // 标记计数
                nms_count--;
                // 标记剔除
                nms_list[i] = -1; 
            }
            // 跳过已经筛选过的box
            if(nms_list[i] != 0)continue;
            // 最大值查询操作
            if(decode_boxes[i*6+5] > max_score){
                max_score = decode_boxes[i*6+5];
                max_class_id = decode_boxes[i*6+4];
                max_index = i;
            }
        }
        // 标记计数
        nms_count--;
        vim_num++;
        // 标记已选
        nms_list[max_index] = 1;
        // 遍历剔除IOU重复项
        for(int i=0; i<dboxes_num; i++){
            // 跳过已经筛选过的box
            if(nms_list[i] != 0)continue;
            // 跳过不同class
            if(round(decode_boxes[i*6+4]) != max_class_id)continue;
            // 计算IOU并标记剔除
            if(iou(&decode_boxes[max_index*6], &decode_boxes[i*6]) > nms_threshold){
                // 标记计数
                nms_count--;
                // 标记剔除
                nms_list[i] = -1;
            }
        }
    }
    return vim_num;
}

void RDKyolov5postprocess::score_filter(
    float* boxes0,
    float* boxes1,
    float* boxes2,
    int begin_index,
    int filter_len
){
    int index = begin_index; // box索引值
    int index_head;          // 索引头
    int loop_count = -1;     // 循环计数变量

    int box_id = 0;          // box id: 0 1 2
    int dbox_id;         
    float* boxes = nullptr;   // 3个box的引用

    int best_cls_id;         // 最大概率类id
    float x, y, w, h, score, best_prob;

    int index_h; // 多维索引值
    int index_w; // 多维索引值
    int index_c; // 多维索引值

    // 仅对filter_len长度处理
    while(++loop_count < filter_len){
        // box_id切换与index索引处理
        while(index >= boxes_num[box_id]){
            index -= boxes_num[box_id++];
            // box选择并引用
        }
        switch(box_id){
            case 0: boxes = boxes0; break;
            case 1: boxes = boxes1; break;
            case 2: boxes = boxes2; break;
            default: out_of_range("The indexes of the 3 boxes are out of range, check whether the parameters are incorrect.");
        }
        
        // 多维数组索引头
        index_head = index*(classes_number+5);
        
        // conf初步筛选
        if(boxes[index_head+4]>=deactivate_score){ 
            // 查询最大prob的cls_id
            best_cls_id = 0;
            best_prob = boxes[index_head + 5 + 0];
            for(int class_id=0; class_id<classes_number; class_id++){
                if(boxes[index_head + 5 + class_id] > best_prob){
                    best_prob = boxes[index_head + 5 + class_id];
                    best_cls_id = class_id;
                }
            }
            // prob初步筛选
            if(best_prob>=deactivate_score){
                // 计算score
                score = activate(boxes[index_head+4]) * activate(best_prob);
                // 得分过滤
                if(score_threshold <= score){ 
                    // yolov5解码
                    // 逆算索引值
                    index_c = index % 3;
                    index_h = (index / 3) % (model_size / stride[box_id]);
                    index_w = (index / 3 / (model_size / stride[box_id])) % (model_size / stride[box_id]);
                    // 解算x,y值
                    x = (activate(boxes[index_head])   * 2 - 0.5 + index_h) * stride[box_id];
                    y = (activate(boxes[index_head+1]) * 2 - 0.5 + index_w) * stride[box_id];
                    // 解算w,h值
                    w = 4 * activate(boxes[index_head+2]) * activate(boxes[index_head+2]) * anchors[box_id][index_c][0];
                    h = 4 * activate(boxes[index_head+3]) * activate(boxes[index_head+3]) * anchors[box_id][index_c][1];
                    // 验证xywh  
                    if(w>0 && h>0){
                        // 使用原子操作获得decode_boxes的索引id
                        dbox_id = __sync_fetch_and_add(&dboxes_num, 1);
                        // 转换到xmin, ymin, xmax, ymax并存入
                        decode_boxes[dbox_id*6]   = max(0.0, x - 0.5 * w);
                        decode_boxes[dbox_id*6+1] = max(0.0, y - 0.5 * h);
                        decode_boxes[dbox_id*6+2] = min(double(model_size), x + 0.5 * w);
                        decode_boxes[dbox_id*6+3] = min(double(model_size), y + 0.5 * h);
                        // 装入class_id与score
                        decode_boxes[dbox_id*6+4] = float(best_cls_id);
                        decode_boxes[dbox_id*6+5] = score;    
                    }         
                }
            }
        }
        // 更新索引值
        index++;
    }
    return ;
}

void RDKyolov5postprocess::get_results(
    float* results
){
    int count = 0;
    for(int i=0; i<dboxes_num; i++){
        if(nms_list[i] != 1)continue;
        for(int j=0; j<6; j++){
            results[count*6+j] = decode_boxes[i*6+j];
            if(j<=3)results[count*6+j]/=model_size;
        }
        count++;
    }
}

RDKyolov5postprocess::~RDKyolov5postprocess(){
    delete[] decode_boxes;
    delete[] nms_list;
}