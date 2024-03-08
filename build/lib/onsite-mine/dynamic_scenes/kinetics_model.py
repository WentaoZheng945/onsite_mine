#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Wentao Zheng
# datetime： 2024/3/4 21:13 
# ide： PyCharm
import os
import sys
import shlex
import platform
import subprocess
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# 自定义库
from dynamic_scenes.socket_module import Client
from dynamic_scenes.observation import Observation

class KineticsModelStarter():
    def __init__(self, observation: Observation):
        initial_state = self._get_init_state_of_ego(observation)
        self.client = Client()
        self._write_temp_script(initial_state[0], initial_state[1], initial_state[2], initial_state[3])
        self._check_completed()
        pass

    @property
    def get_client(self):
        return self.client

    def _get_init_state_of_ego(self, observation:Observation):
        x = observation.vehicle_info['ego']['x']
        y = observation.vehicle_info['ego']['y']
        yaw = observation.vehicle_info['ego']['yaw_rad']
        v0 = observation.vehicle_info['ego']['v_mps']
        return x,y,yaw,v0

    def _write_temp_script(self, x:float, y:float, yaw:float, v0:float):
        # 编写一个tempScript.m脚本用于存储初始化信息
        current_path = Path(__file__).parent
        os_type = self._judge_platform()
        tempscript = current_path.parent / 'kinetic_model' / f'{os_type}'/'tempScript.m'
        with open(tempscript, 'w') as f:
            f.write(f"x0={x};\n")
            f.write(f"y0={y};\n")
            f.write(f"head={yaw};\n")
            f.write(f"v0={v0};\n")
            f.write("acc=0.0;\n")  # 初始加速度
            f.write("gear=2;\n")  # 初始档位：1-前进档；2-驻车档；3-倒车档
            f.write("yaw=0.0;\n")  # 初始前轮转角
            f.write("load('a_brake.mat');\n")
            f.write("load('a_thr.mat');\n")
            f.write("load('brake.mat');\n")
            f.write("load('thr.mat');\n")
            f.write("modelName='VehicleModel';\n")
            f.write("run('control_simulink_with_udp.m');\n")

        command = f"matlab -r \"run('{tempscript.as_posix()}')\""
        result = subprocess.Popen(shlex.split(command))
        return result

    def _check_completed(self):
        # Check whether the initialization is complete
        data, _ = self.client.client_receive_sock.recvfrom(1024)  # 假设信号很小，不需要大缓冲区
        if data.decode() == 'ready':
            print("MATLAB就绪，继续执行")

    def _judge_platform(self):
        os_type = platform.system()
        if os_type == "Windows":
            return 'win'
        elif os_type == "Linux" or os_type == "Darwin":
            return 'linux'
        else:
            print(f"不支持的操作系统: {os_type}")