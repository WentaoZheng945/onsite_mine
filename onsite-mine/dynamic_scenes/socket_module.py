# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2024/2/8 12:17
import socket
import struct  # 解析simulink模型打包来的数据要用
import time

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
        message = struct.pack('ddd', steering_angle, gear, acceleration)
        self.client_send_sock.sendto(message, (self.send_ip, self.send_port))
        # time.sleep(0.001)
        print('发送成功！！！')

        data, addr = self.client_receive_sock.recvfrom(56)
        unpacked_data = struct.unpack('ddddddd', data)

        phi = unpacked_data[0]  # 航向角
        X = unpacked_data[1]  # 全局X
        Y = unpacked_data[2]  # 全局Y
        dx = unpacked_data[3]  # x向速度
        dy = unpacked_data[4]  # y向速度
        dphi = unpacked_data[5]  # 角速度
        time = unpacked_data[6]  # 仿真时间（simulink中）
        print('接收成功！！！')
        return unpacked_data





