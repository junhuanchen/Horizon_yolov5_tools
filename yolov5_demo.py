# ==================================================
#
#     yolov5测试代码
#       * Author: Leaf
#       * Date: 2024-01-26
#
# ==================================================
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from fast_postprocess import RDKyolov5postprocess
import bpu_yolov5_tools as tools  
import argparse, yaml, time, cv2, os

if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/app/model/basic/yolov5s_672x672_nv12.bin', help="bin模型路径")
    parser.add_argument('--image', type=str, default='./images/kite.jpg', help='推理图片路径')
    parser.add_argument('--names', type=str, default='./models/demo.yaml', help='name列表yaml文件路径')
    parser.add_argument('--sco_thr', type=float, default=0.4, help='得分阈值')
    parser.add_argument('--nms_thr', type=float, default=0.45, help='NMS阈值')
    parser.add_argument('--thread_num', type=int, default=8, help='后处理线程数')
    parser.add_argument('--nice', type=int, default=0, help='程序优先级')
    parser.add_argument('--CPUOC', type=bool, default=True, help='是否CPU超频')
    opt = parser.parse_args()

    try:
        os.system("sudo bash -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'")

        # 调整程序优先级
        os.nice(opt.nice)

        # 开启cpu超频
        if opt.CPUOC:
            os.system("sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpufreq/boost'")

        # 加载模型
        models = dnn.load(opt.model)

        # cv2加载图片
        img = cv2.imread(opt.image) 

        # 加载names.yaml文件
        with open(opt.names, 'r') as file: 
            names = yaml.safe_load(file)

        # 显示模型信息并验证模型格式
        tools.models_info_show(models) 
        if tools.model_val(models) == 0:
            print('模型验证通过！')
        else:
            raise ValueError('模型格式错误！')

        # 获取模型信息
        model_size = models[0].inputs[0].properties.shape[2]
        class_number = models[0].outputs[0].properties.shape[-1]//3-5
        print("模型信息", model_size, class_number)
        # 后处理工具
        pst = RDKyolov5postprocess.yolov5postprocess(
            model_size, 
            class_number, 
            score_threshold = opt.sco_thr, 
            nms_threshold = opt.nms_thr, 
            thread_num = opt.thread_num)

        # 图片格式转换
        h, w = models[0].inputs[0].properties.shape[2:]
        resized_data = cv2.resize(img, (h,w), interpolation=cv2.INTER_AREA)
        nv12_data = tools.bgr2nv12(resized_data)

        t0 = time.time()

        # 模型推理
        outputs = models[0].forward(nv12_data)
        t1 = time.time()

        # 后处理
        # 如果需要测试后处理1k次，可以把下方注释解除
        # # for i in range(100):
        #     outputs = models[0].forward(nv12_data)
        #     results = pst.process(outputs)
        results = pst.process(outputs)
        t2 = time.time()

        results[:,0] *= img.shape[1]
        results[:,1] *= img.shape[0]
        results[:,2] *= img.shape[1]
        results[:,3] *= img.shape[0]

        # 输出结果
        tools.result_show(results, names['names'])

        # 输出时间
        print("infer time:",1000*(t1-t0),"ms")
        print("postprocess time:",1000*(t2-t1),"ms")

        # 可视化绘制
        img = tools.result_draw(img, results, names['names'])

        # 保存图片
        cv2.imwrite('results/'+opt.image.split('/')[-1][0:-4]+'.jpg', img)

    finally:
        # 关闭CPU超频
        os.system("sudo bash -c 'echo 0 > /sys/devices/system/cpu/cpufreq/boost'")
        # 恢复默认的任务优先极
        os.nice(-opt.nice)

