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
import argparse, yaml, time, os
import asyncio
import nats
import sysv_ipc

async def main():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/root/X3_Media/demo/cpp_nats_pub_img_with_face/Horizon_yolov5_tools/models/lwh_yolov5s_672x672_nv12.bin', help="bin模型路径")
    parser.add_argument('--names', type=str, default='./models/lwh_demo.yaml', help='name列表yaml文件路径')
    parser.add_argument('--sco_thr', type=float, default=0.3, help='得分阈值')
    parser.add_argument('--nms_thr', type=float, default=0.35, help='NMS阈值')
    parser.add_argument('--thread_num', type=int, default=2, help='后处理线程数')
    parser.add_argument('--nice', type=int, default=0, help='程序优先级')
    parser.add_argument('--CPUOC', type=bool, default=True, help='是否CPU超频')
    opt = parser.parse_args()

    # PY 的接口会内存泄漏，不影响应用，可以优化，或设计成 泄漏到多少后自动退出重开也可以解决，程序重入的时间大概 最快 400 - 最慢 500ms 所以也可以用。

    try:
        os.system("sudo bash -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'")

        # 调整程序优先级
        os.nice(opt.nice)

        # 开启cpu超频
        if opt.CPUOC:
            os.system("sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpufreq/boost'")

        # 加载模型
        models = dnn.load(opt.model)

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

        # 连接到 NATS 服务器
        nc = await nats.connect(servers=["nats://localhost:4222"])

        h, w = models[0].inputs[0].properties.shape[2:]
        shared_mem = sysv_ipc.SharedMemory(0x1234, flags=sysv_ipc.IPC_CREAT, size=int(h*w*1.5))
        nv12_data = np.frombuffer(shared_mem, dtype=np.uint8) # 内存映射有点特殊

        while True:
            # nv12_data = np.frombuffer(shared_mem, dtype=np.uint8)
            
            t0 = time.time()
            
            # 模型推理
            outputs = models[0].forward(nv12_data)
            
            t1 = time.time()

            # 后处理
            results = pst.process(outputs)

            results[:,0] *= w
            results[:,1] *= h
            results[:,2] *= w
            results[:,3] *= h

            t2 = time.time()

            # 格式化输出结果
            # if len(results):
            #     tools.result_show(results, names['names'])

            # 获取结果
            for result in results:
                bbox = result[:4] # 矩形框位置信息  
                score = result[5] # 置信度得分
                id_number = int(result[4]) # id
                name = names['names'][id_number] # 名称
                # print('{:<9.2f}{:<9.2f}{:<9.2f}{:<9.2f}{:<9d}{:<16}{:<9.2f}'.format(bbox[0], bbox[1], bbox[2], bbox[3], id_number, name, score))

                # 根据 JSON 模板格式化结果
                json_template = R'{"data":"%s","top_left_x":%d,"top_left_y":%d,"bottom_right_x":%d,"bottom_right_y":%d}' % (
                    name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                )
                # 发布到 NATS
                await nc.publish("app.vision.yolo", json_template.encode('utf-8'))
                # print(json_template)

            # 输出时间
            # print("infer time:",1000*(t1-t0),"ms")
            # print("postprocess time:",1000*(t2-t1),"ms")
            await nc.flush()

    except Exception as e:
        print("Exception occurred: ", str(e))
    finally:
        # 关闭CPU超频
        os.system("sudo bash -c 'echo 0 > /sys/devices/system/cpu/cpufreq/boost'")
        # 恢复默认的任务优先极
        os.nice(-opt.nice)
        for tmp in models:
            del tmp

if __name__ == '__main__':
    asyncio.run(main())
