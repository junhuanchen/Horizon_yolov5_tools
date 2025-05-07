import subprocess
import time
import argparse
import os

def get_memory_usage(pid):
    """
    通过读取 /proc/<pid>/status 文件获取进程的内存使用情况（单位：MB）
    """
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    memory_usage_kb = int(line.split(":")[1].strip().split(" ")[0])
                    memory_usage_mb = memory_usage_kb / 1024
                    return memory_usage_mb
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"获取内存使用信息时发生错误：{e}")
        return None

def monitor_script(script_path, memory_threshold_mb, check_interval):
    """
    监控并重启外部 Python 脚本，确保每次只有一个实例运行，但在终止当前实例之前先启动另一个实例

    :param script_path: 外部脚本的路径
    :param memory_threshold_mb: 内存使用阈值（单位：MB）
    :param check_interval: 检查间隔时间（单位：秒）
    """
    current_process = None  # 当前运行的实例

    while True:
        if current_process is None or current_process.poll() is not None:
            # 如果当前实例为空或已结束，则启动新实例
            current_process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"启动新实例，进程ID：{current_process.pid}")

        try:
            while current_process.poll() is None:  # 检查当前实例是否仍在运行
                memory_usage = get_memory_usage(current_process.pid)
                if memory_usage is not None:
                    print(f"当前实例内存使用：{memory_usage:.2f} MB")

                    # 如果内存使用超过阈值，启动新实例并终止当前实例
                    if memory_usage > memory_threshold_mb:
                        print(f"内存使用超过阈值 {memory_threshold_mb} MB，启动新实例并终止当前实例...")
                        
                        # 启动新实例
                        new_process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"启动新实例，进程ID：{new_process.pid}")
                        
                        # 等待3秒后终止当前实例
                        time.sleep(3)
                        current_process.terminate()
                        current_process.wait()
                        print(f"终止当前实例，进程ID：{current_process.pid}")
                        
                        # 更新当前实例为新实例
                        current_process = new_process
                        break
                else:
                    print("无法获取内存使用信息，跳过本次检查")
                
                # 等待一段时间后再次检查
                time.sleep(check_interval)
        except Exception as e:
            print(f"发生错误：{e}")
            if current_process:
                current_process.terminate()
                current_process.wait()
            current_process = None

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="监控并重启外部 Python 脚本")
    parser.add_argument("script_path", type=str, help="外部脚本的路径")
    parser.add_argument("--memory_threshold_mb", type=float, default=128, help="内存使用阈值（单位：MB，默认128MB）")
    parser.add_argument("--check_interval", type=int, default=10, help="检查间隔时间（单位：秒，默认10秒）")
    args = parser.parse_args()

    # 调用监控函数
    monitor_script(args.script_path, args.memory_threshold_mb, args.check_interval)
    # python dls_monitor.py yolov5_shard_mem.py --memory_threshold_mb 256 --check_interval 1

