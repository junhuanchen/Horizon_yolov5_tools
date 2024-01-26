/*
==================================================

  yolov5 BPU部署模型 快速后处理 C++实现

    activate: 激活函数（快速sigmoid）

    iou: IOU计算函数

    nms: 非极大值抑制函数

    decoder_son: yolov5解码子函数
    *本处对一组yolov5数据（长度为5+class_num）进
    行解码，优先计算得分值并筛选，避免了巨量凸余的
    激活运算。

    decoder: yolov5解码函数
    *原本是为了在yolov5解码内部加入多线程优化，才
    将子函数独立，但效果不佳，处理不当容易内存溢出
    和线程间竞争，所以取消在此处多线程，但该子函数
    保留并内联。

    fast_postprocess: 后处理统合函数
    *将传入的3个output分别切片处理成4*4 2*2和1*1
    的box片段，再开启21个线程同时进行解码，最后统
    合输出。

    大概已经接近C++做后处理的极限了，如有优化建议
    欢迎戳邮箱：zxy_yys_leaf@163.com

    * Author: Leaf
    * Date: 2024-01-26
     
==================================================
*/

#include <iostream>
#include <thread>
#include <cmath>

using namespace std;

// 全局定义reshape和transpose的多维索引
#define index_5(box_size, class_num, i0, i1, i2, i3, i4) ((i0) * (box_size) * (box_size) * 3 * ((class_num) + 5) + (i1) * (box_size) * 3 * ((class_num) + 5) + (i2) * 3 * ((class_num) + 5) + (i3) * ((class_num) + 5) + (i4))
#define transpose_index_5(box_size, class_num, i0, i1, i2, i3, i4) index_5(box_size, class_num, i0, i2, i3, i1, i4)

inline float activate(
    float x)
{
    // fats sigmoid
    return 0.5 * (1 + tanh(0.5 * x));
}

// 计算IOU（交并比）
inline float iou(
    float *boxes1,
    float *boxes2)
{

    float // 计算两个边界框的最小和最大坐标
        left = max(boxes1[0], boxes2[0]),
        top = max(boxes1[1], boxes2[1]),
        right = min(boxes1[2], boxes2[2]),
        bottom = min(boxes1[3], boxes2[3]);

    float // 计算交集区域的宽度和高度
        int_w = max(float(0), right - left),
        int_h = max(float(0), bottom - top);

    if (int_w == 0 || int_h == 0){
        
        return 0.0; // 如果没有交集，直接返回0
    }

    float int_area = int_w * int_h; // 计算交集面积

    float // 计算两个边界框的面积
        area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1]),
        area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]);

    float uni_area = area1 + area2 - int_area; // 计算并集面积

    if (uni_area == 0.0) return 0.0; // 防止除以0

    return int_area / uni_area; // 计算交并比
}

// 非极大值抑制
inline void nms(
    /*const*/ float *box,
    int *best_box,
    const int box_lenth,
    const float nms_threshold,
    int *bbox_lenth)
{

    *bbox_lenth = 0;                        // 初值
    bool *bool_box = new bool[box_lenth]{}; // 初始化为false，表示未访问box
    int remain_lenth = box_lenth;           // 剩余box数量

    while (remain_lenth > 0) // 还存在未访问box时循环遍历
    {
        int class_id = -1;  // 本轮筛选的目标类id，初值为-1
        int max_index = -1; // 对应索引，初值为-1
        float max_score;    // 对应得分

        // 遍历搜索，使第一个可用的box类为目标类，并搜索的得分最高的这一类别
        for (int i = 0; i < box_lenth; i++)
        {
            if (bool_box[i])
                continue; // 该box已访问则跳过
            if (class_id == -1)
            {
                // 比较初值
                class_id = (int)box[i * 6 + 4]; // 确认目标类别
                max_index = i;                  // 最大得分索引
                max_score = box[i * 6 + 5];     // 最大得分值
            }
            else if (class_id == box[i * 6 + 4] && max_score < box[i * 6 + 5]) // 得分更高的目标class_id
            {
                // 替换
                max_score = box[i * 6 + 5];
                max_index = i;
            }
        }

        bool_box[max_index] = true; // 标记已访问
        remain_lenth--;             // 剩余长度-1

        // 当且仅当此时会筛选目标到best_box
        best_box[(*bbox_lenth)++] = max_index; // 加入best_box并且索引自增

        // 再次遍历，根据iou筛选
        for (int i = 0; i < box_lenth; i++)
        {
            if (bool_box[i] || box[i * 6 + 4] != class_id)
                continue; // 如果是 已访问or非同类id 则跳过

            if (iou(&box[i * 6], &box[max_index * 6]) >= nms_threshold) // iou计算并筛选阈值
            {
                // 标记已访问但不加入best_box
                bool_box[i] = true;
                remain_lenth--;
            }
        }
    }
    delete[] bool_box; // 释放动态内存
    return;
}

// 解码子函数
inline void decoder_son(
    float *output,
    float *box,
    int index,
    int box_size,
    float score_threshold,
    int stride,
    int anchor_i1_0,
    int anchor_i1_1,
    int i2,
    int i3,
    bool *ret)
{

    int max_index;
    float max_conf;
    float score;
    float x, y, w, h, xmin, ymin, xmax, ymax;
    int origin_size = stride * box_size - 1;

    max_index = 5;
    max_conf = output[5];

    for (int i4 = 5; i4 < 85; i4++)
    {
        if (max_conf < output[i4])
        {
            max_conf = output[i4];
            max_index = i4;
        }
    }

    score = activate(max_conf) * activate(output[4]); // 计算得分

    if (score >= score_threshold)
    {
        // 计算并储存数据
        // 计算coor
        x = (activate(output[0]) * 2.0 - 0.5 + i3) * stride;
        y = (activate(output[1]) * 2.0 - 0.5 + i2) * stride;
        w = pow(2 * activate(output[2]), 2) * anchor_i1_0;
        h = pow(2 * activate(output[3]), 2) * anchor_i1_1;


        // 剔除h和w异常的数据
        if (w <= 0 || h <= 0)
        {
            *ret = 0;
            return;
        }

        // 转换coor格式
        xmin = x - w / 2;
        ymin = y - h / 2;
        xmax = x + w / 2;
        ymax = y + h / 2;

        // 修正coor
        if (xmin < 0)
            xmin = 0;
        if (ymin < 0)
            ymin = 0;
        if (xmax > origin_size)
            xmax = origin_size;
        if (ymax > origin_size)
            ymax = origin_size;

        // 剔除异常coor
        if (xmax - xmin <= 0 || ymax - ymin <= 0)
        {
            *ret = 0;
            return;
        }

        box[index*6 + 0] = xmin;
        box[index*6 + 1] = ymin;
        box[index*6 + 2] = xmax;
        box[index*6 + 3] = ymax;

        box[index*6 + 4] = float(max_index) - 5.0; // class ID
        box[index*6 + 5] = score;                  // 得分数据

        *ret = 1;
        return;
    }
    else
    {
        *ret = 0;
        return;
    }
}

inline void decoder(
    float *output,
    int begin_idx2,
    int begin_idx3,
    int finish_idx2,
    int finish_idx3,
    float *box,
    int box_size,
    int class_number,
    float score_threshold,
    int anchor[3][2],
    int stride,
    int middle_data_long,
    int *data_len)
{
    int middle_data_count = 0;
    bool ret;

    // 索引0
    #define index0 0

    // 索引1
    for (int i1 = 0; i1 < 3; i1++)
    {
        int index1 = index0 + transpose_index_5(box_size, class_number, 0, i1, 0, 0, 0);

        // 索引2
        for (int i2 = begin_idx2; i2 < finish_idx2; i2++)
        {
            int index2 = index1 + transpose_index_5(box_size, class_number, 0, 0, i2, 0, 0);

            // 索引3
            for (int i3 = begin_idx3; i3 < finish_idx3; i3++)
            {
                int index3 = index2 + transpose_index_5(box_size, class_number, 0, 0, 0, i3, 0);

                // 对索引4进行操作

                decoder_son(&output[index3], &box[0], middle_data_count, box_size, score_threshold, stride, anchor[i1][0], anchor[i1][1], i2, i3, &ret);
                middle_data_count += ret;
                

                // 中间数据计数，防止内存泄漏，如果发生则终止程序

                if (middle_data_count >= middle_data_long)
                {
                    cout << "[ERROR]postprocess.cpp: The middle data(size of " << box_size << ") length out of index!!!" << endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    (*data_len) = middle_data_count;
    return;
}

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
    int box_number)
{
    int anchors[3][3][2] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    int strides[3] = {8, 16, 32};

    float // 为多线程所有box分配空间
        *box0_0_0 = new float[middle_data_long * 6],
        *box0_0_1 = new float[middle_data_long * 6],
        *box0_0_2 = new float[middle_data_long * 6],
        *box0_0_3 = new float[middle_data_long * 6],

        *box0_1_0 = new float[middle_data_long * 6],
        *box0_1_1 = new float[middle_data_long * 6],
        *box0_1_2 = new float[middle_data_long * 6],
        *box0_1_3 = new float[middle_data_long * 6],

        *box0_2_0 = new float[middle_data_long * 6],
        *box0_2_1 = new float[middle_data_long * 6],
        *box0_2_2 = new float[middle_data_long * 6],
        *box0_2_3 = new float[middle_data_long * 6],

        *box0_3_0 = new float[middle_data_long * 6],
        *box0_3_1 = new float[middle_data_long * 6],
        *box0_3_2 = new float[middle_data_long * 6],
        *box0_3_3 = new float[middle_data_long * 6],

        *box1_0_0 = new float[middle_data_long * 6],
        *box1_0_1 = new float[middle_data_long * 6],

        *box1_1_0 = new float[middle_data_long * 6],
        *box1_1_1 = new float[middle_data_long * 6],

        *box2_0_0 = new float[middle_data_long * 6];

    // 宏定义数组索引段
    #define par_0_0 (0 * model_size / strides[0] / 4)
    #define par_0_1 (1 * model_size / strides[0] / 4)
    #define par_0_2 (2 * model_size / strides[0] / 4)
    #define par_0_3 (3 * model_size / strides[0] / 4)
    #define par_0_4 (4 * model_size / strides[0] / 4)

    #define par_1_0 (0 * model_size / strides[1] / 2)
    #define par_1_1 (1 * model_size / strides[1] / 2)
    #define par_1_2 (2 * model_size / strides[1] / 2)

    #define par_2_0 0
    #define par_2_1 model_size / strides[2]

    int // 声明每个线程计算得到的数据长度
        data_num0_0_0,
        data_num0_0_1,
        data_num0_0_2,
        data_num0_0_3,

        data_num0_1_0,
        data_num0_1_1,
        data_num0_1_2,
        data_num0_1_3,

        data_num0_2_0,
        data_num0_2_1,
        data_num0_2_2,
        data_num0_2_3,

        data_num0_3_0,
        data_num0_3_1,
        data_num0_3_2,
        data_num0_3_3,

        data_num1_0_0,
        data_num1_0_1,

        data_num1_1_0,
        data_num1_1_1,

        data_num2_0_0;

    int 
    par_size0 = model_size/8,
    par_size1 = model_size/16,
    par_size2 = model_size/32;

    thread // 21线程并行解码
        decoder0_0_0(decoder, &output0[0], par_0_0, par_0_0, par_0_1, par_0_1, &box0_0_0[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_0_0),
        decoder0_0_1(decoder, &output0[0], par_0_0, par_0_1, par_0_1, par_0_2, &box0_0_1[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_0_1),
        decoder0_0_2(decoder, &output0[0], par_0_0, par_0_2, par_0_1, par_0_3, &box0_0_2[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_0_2),
        decoder0_0_3(decoder, &output0[0], par_0_0, par_0_3, par_0_1, par_0_4, &box0_0_3[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_0_3),

        decoder0_1_0(decoder, &output0[0], par_0_1, par_0_0, par_0_2, par_0_1, &box0_1_0[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_1_0),
        decoder0_1_1(decoder, &output0[0], par_0_1, par_0_1, par_0_2, par_0_2, &box0_1_1[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_1_1),
        decoder0_1_2(decoder, &output0[0], par_0_1, par_0_2, par_0_2, par_0_3, &box0_1_2[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_1_2),
        decoder0_1_3(decoder, &output0[0], par_0_1, par_0_3, par_0_2, par_0_4, &box0_1_3[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_1_3),

        decoder0_2_0(decoder, &output0[0], par_0_2, par_0_0, par_0_3, par_0_1, &box0_2_0[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_2_0),
        decoder0_2_1(decoder, &output0[0], par_0_2, par_0_1, par_0_3, par_0_2, &box0_2_1[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_2_1),
        decoder0_2_2(decoder, &output0[0], par_0_2, par_0_2, par_0_3, par_0_3, &box0_2_2[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_2_2),
        decoder0_2_3(decoder, &output0[0], par_0_2, par_0_3, par_0_3, par_0_4, &box0_2_3[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_2_3),

        decoder0_3_0(decoder, &output0[0], par_0_3, par_0_0, par_0_4, par_0_1, &box0_3_0[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_3_0),
        decoder0_3_1(decoder, &output0[0], par_0_3, par_0_1, par_0_4, par_0_2, &box0_3_1[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_3_1),
        decoder0_3_2(decoder, &output0[0], par_0_3, par_0_2, par_0_4, par_0_3, &box0_3_2[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_3_2),
        decoder0_3_3(decoder, &output0[0], par_0_3, par_0_3, par_0_4, par_0_4, &box0_3_3[0], par_size0, class_number, score_threshold, anchors[0], strides[0], middle_data_long, &data_num0_3_3),

        decoder1_0_0(decoder, &output1[0], par_1_0, par_1_0, par_1_1, par_1_1, &box1_0_0[0], par_size1, class_number, score_threshold, anchors[1], strides[1], middle_data_long, &data_num1_0_0),
        decoder1_0_1(decoder, &output1[0], par_1_0, par_1_1, par_1_1, par_1_2, &box1_0_1[0], par_size1, class_number, score_threshold, anchors[1], strides[1], middle_data_long, &data_num1_0_1),

        decoder1_1_0(decoder, &output1[0], par_1_1, par_1_0, par_1_2, par_1_1, &box1_1_0[0], par_size1, class_number, score_threshold, anchors[1], strides[1], middle_data_long, &data_num1_1_0),
        decoder1_1_1(decoder, &output1[0], par_1_1, par_1_1, par_1_2, par_1_2, &box1_1_1[0], par_size1, class_number, score_threshold, anchors[1], strides[1], middle_data_long, &data_num1_1_1),

        decoder2_0_0(decoder, &output2[0], par_2_0, par_2_0, par_2_1, par_2_1, &box2_0_0[0], par_size2, class_number, score_threshold, anchors[2], strides[2], middle_data_long, &data_num2_0_0);

    // 等到所有线程结束
    decoder0_0_0.join();
    decoder0_0_1.join();
    decoder0_0_2.join();
    decoder0_0_3.join();

    decoder0_1_0.join();
    decoder0_1_1.join();
    decoder0_1_2.join();
    decoder0_1_3.join();

    decoder0_2_0.join();
    decoder0_2_1.join();
    decoder0_2_2.join();
    decoder0_2_3.join();

    decoder0_3_0.join();
    decoder0_3_1.join();
    decoder0_3_2.join();
    decoder0_3_3.join();

    decoder1_0_0.join();
    decoder1_0_1.join();

    decoder1_1_0.join();
    decoder1_1_1.join();

    decoder2_0_0.join();

    // 统合数据
    int data_num =
        data_num0_0_0 +
        data_num0_0_1 +
        data_num0_0_2 +
        data_num0_0_3 +

        data_num0_1_0 +
        data_num0_1_1 +
        data_num0_1_2 +
        data_num0_1_3 +

        data_num0_2_0 +
        data_num0_2_1 +
        data_num0_2_2 +
        data_num0_2_3 +

        data_num0_3_0 +
        data_num0_3_1 +
        data_num0_3_2 +
        data_num0_3_3 +

        data_num1_0_0 +
        data_num1_0_1 +

        data_num1_1_0 +
        data_num1_1_1 +

        data_num2_0_0;

    // 准备新的容器
    float *pred_box = new float[data_num * 6];

    for (int j = 0; j < 6; j++)
    {
        int data_count = 0;
        for (int i = 0; i < data_num0_0_0; i++)
            pred_box[(data_count++) * 6 + j] = box0_0_0[i * 6 + j];
        for (int i = 0; i < data_num0_0_1; i++)
            pred_box[(data_count++) * 6 + j] = box0_0_1[i * 6 + j];
        for (int i = 0; i < data_num0_0_2; i++)
            pred_box[(data_count++) * 6 + j] = box0_0_2[i * 6 + j];
        for (int i = 0; i < data_num0_0_3; i++)
            pred_box[(data_count++) * 6 + j] = box0_0_3[i * 6 + j];

        for (int i = 0; i < data_num0_1_0; i++)
            pred_box[(data_count++) * 6 + j] = box0_1_0[i * 6 + j];
        for (int i = 0; i < data_num0_1_1; i++)
            pred_box[(data_count++) * 6 + j] = box0_1_1[i * 6 + j];
        for (int i = 0; i < data_num0_1_2; i++)
            pred_box[(data_count++) * 6 + j] = box0_1_2[i * 6 + j];
        for (int i = 0; i < data_num0_1_3; i++)
            pred_box[(data_count++) * 6 + j] = box0_1_3[i * 6 + j];

        for (int i = 0; i < data_num0_2_0; i++)
            pred_box[(data_count++) * 6 + j] = box0_2_0[i * 6 + j];
        for (int i = 0; i < data_num0_2_1; i++)
            pred_box[(data_count++) * 6 + j] = box0_2_1[i * 6 + j];
        for (int i = 0; i < data_num0_2_2; i++)
            pred_box[(data_count++) * 6 + j] = box0_2_2[i * 6 + j];
        for (int i = 0; i < data_num0_2_3; i++)
            pred_box[(data_count++) * 6 + j] = box0_2_3[i * 6 + j];

        for (int i = 0; i < data_num0_3_0; i++)
            pred_box[(data_count++) * 6 + j] = box0_3_0[i * 6 + j];
        for (int i = 0; i < data_num0_3_1; i++)
            pred_box[(data_count++) * 6 + j] = box0_3_1[i * 6 + j];
        for (int i = 0; i < data_num0_3_2; i++)
            pred_box[(data_count++) * 6 + j] = box0_3_2[i * 6 + j];
        for (int i = 0; i < data_num0_3_3; i++)
            pred_box[(data_count++) * 6 + j] = box0_3_3[i * 6 + j];

        for (int i = 0; i < data_num1_0_0; i++)
            pred_box[(data_count++) * 6 + j] = box1_0_0[i * 6 + j];
        for (int i = 0; i < data_num1_0_1; i++)
            pred_box[(data_count++) * 6 + j] = box1_0_1[i * 6 + j];

        for (int i = 0; i < data_num1_1_0; i++)
            pred_box[(data_count++) * 6 + j] = box1_1_0[i * 6 + j];
        for (int i = 0; i < data_num1_1_1; i++)
            pred_box[(data_count++) * 6 + j] = box1_1_1[i * 6 + j];

        for (int i = 0; i < data_num2_0_0; i++)
            pred_box[(data_count++) * 6 + j] = box2_0_0[i * 6 + j];

        if (data_count != data_num)
        {
            // 除非内存问题，否则此处正常不会报错
            cout << "[ERROR]postprocess.cpp: The all pred_box data(size of " << data_count << ") length error!!!" << endl;
            exit(EXIT_FAILURE);
        }
    }

    // 释放临时空间
    delete[] box0_0_0;
    delete[] box0_0_1;
    delete[] box0_0_2;
    delete[] box0_0_3;

    delete[] box0_1_0;
    delete[] box0_1_1;
    delete[] box0_1_2;
    delete[] box0_1_3;

    delete[] box0_2_0;
    delete[] box0_2_1;
    delete[] box0_2_2;
    delete[] box0_2_3;

    delete[] box0_3_0;
    delete[] box0_3_1;
    delete[] box0_3_2;
    delete[] box0_3_3;

    delete[] box1_0_0;
    delete[] box1_0_1;

    delete[] box1_1_0;
    delete[] box1_1_1;

    delete[] box2_0_0;

    int *best_box_index = new int[data_num]{}; // best_box数组索引
    int best_box_lenth; // best_box数组长度

    // NMS处理
    nms(&pred_box[0], &best_box_index[0], data_num, nms_threshold, &best_box_lenth);

    // best_box容器
    for(int i=0; i<best_box_lenth; i++){
        for(int j=0; j<6; j++){
            best_box[i*6+j] = pred_box[best_box_index[i]*6 + j];
        }
        if(i >= box_number)break;
    }

    // 清理容器内粗
    delete[] pred_box;
    delete[] best_box_index;


    return best_box_lenth;
}
