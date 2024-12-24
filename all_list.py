import numpy as np
import os
import subprocess
import time
from G2 import *
from G3 import *
from G4 import *
from G5 import *


def run_experiments_sequence(script_list, delay=5):
    for script in script_list:
        try:
            print(f"\n开始执行: {script}")
            start_time = time.time()

            # 根据脚本名称调用相应的代码
            if script == 'G2.py':
                run_G2()  # 假设G2.py中有一个run_G2()函数
            elif script == 'G3.py':
                run_G3()  # 假设G3.py中有一个run_G3()函数
            elif script == 'G4.py':
                run_G4()  # 假设G4.py中有一个run_G4()函数
            elif script == 'G5.py':
                run_G5()  # 假设G5.py中有一个run_G5()函数

            execution_time = time.time() - start_time
            print(f"{script} 执行完成")
            print(f"执行时间: {execution_time:.2f} 秒")

            if script != script_list[-1]:
                print(f"等待 {delay} 秒后执行下一个脚本...\n")
                time.sleep(delay)

        except Exception as e:
            print(f"执行 {script} 时发生错误: {str(e)}")
            choice = input("是否继续执行后续脚本? (y/n): ")
            if choice.lower() != 'y':
                break


# 主代码
if __name__ == "__main__":
    # 定义要执行的脚本列表
    scripts = ['G2.py', 'G3.py', 'G4.py', 'G5.py']

    # 执行脚本序列
    run_experiments_sequence(scripts)