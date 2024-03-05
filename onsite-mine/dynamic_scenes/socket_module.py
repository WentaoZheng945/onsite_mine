# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2024/2/8 12:17
import socket
import struct  # 解析simulink模型打包来的数据要用
import time
import platform
import subprocess

class Client():
    def __init__(self, Send_IP='127.0.0.1', Send_Port=25001, Receive_IP='127.0.0.1', Receive_Port=25000):
        self.send_ip = Send_IP
        self.send_port = Send_Port
        self.receive_ip = Receive_IP
        self.receive_port = Receive_Port
        self._build_client()

    def _build_client(self):
        # 发送端
        self.client_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 接收端
        self.client_receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_receive_sock.bind((self.receive_ip, self.receive_port))

    def send_and_receive(self, gear, acceleration, steering_angle):
        try:
            message = struct.pack('>ddd', gear, acceleration, steering_angle)
            self.client_send_sock.sendto(message, (self.send_ip, self.send_port))
            print('控制量已发送，等到接收下一个状态量...')

            data, addr = self.client_receive_sock.recvfrom(1024)
            print('状态量已接收，等待计算下一个控制量...')
            if data:
                unpacked_data = struct.unpack('>dddd', data)
                return unpacked_data
        except Exception as e:
            print(f"通信出现问题！，具体原因为{e}")

    def close_sockets(self):
        self.client_send_sock.close()
        self.client_receive_sock.close()
        self.kill_matlab_processes()

    @staticmethod
    def kill_matlab_processes():
        # 获取当前操作系统类型
        os_type = platform.system()

        try:
            if os_type == "Windows":
                # 对于Windows系统，使用taskkill
                subprocess.run(["taskkill", "/F", "/IM", "MATLAB.exe"], check=True)
                print("所有MATLAB进程已被成功终止。")
            elif os_type == "Linux" or os_type == "Darwin":
                # 对于Linux和MacOS系统，使用pkill
                # MacOS系统也被视为类Unix系统，通常使用和Linux相同的命令
                subprocess.run(["pkill", "matlab"], check=True)
                print("所有MATLAB进程已被成功终止。")
            else:
                print(f"不支持的操作系统: {os_type}")
        except subprocess.CalledProcessError as e:
            print(f"终止MATLAB进程时发生错误：{e}")





