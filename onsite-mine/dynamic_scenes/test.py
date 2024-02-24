#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Wentao Zheng
# datetime： 2024/1/5 12:09 
# ide： PyCharm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
dir_current_file = os.path.dirname(__file__)  # 'D:\documents\onsite项目\onsite-mine非结构赛题\发布\onsite_mine'
print(dir_current_file)
from common import utils
front_left_corner,front_right_corner,rear_left_corner,rear_right_corner = utils.calculate_vehicle_corners(5.416,1.947,2.708,
                                                                                                          2.708,944.558,2101.612,
                                                                                                          1.708)
# [[[1,1], [2,2], [3, 3], [4, 4]], []]
# [[1,1], [2,2], [3, 3], [4, 4]]

def plot_polygon(polygon_list):
    num = len(polygon_list)
    # 创建一个图和一个轴
    fig, ax = plt.subplots()

    # 设置背景色
    fig.patch.set_facecolor('grey')
    ax.set_facecolor('grey')

    for value in polygon_list:
        polygon_value = patches.Polygon(value, closed=True, fill=True, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(polygon_value)

    # 设置坐标轴的范围
    x_list = []
    y_list = []
    for i in polygon_list:
        x_list_temp, y_list_temp = zip(*i)
        x_list += x_list_temp
        y_list += y_list_temp
    ax.set_xlim(min(x_list)-10, max(x_list)+10)
    ax.set_ylim(min(y_list)-10, max(y_list)+10)

    # 显示图表
    plt.show()
# print([[front_left_corner,front_right_corner,rear_left_corner,rear_right_corner]])
polygon_list_ = [[[943.223265706659, 2104.161402011488], [945.1519684478187, 2104.427700220726], [945.8927342933409, 2099.062597988512], [943.9640315521813, 2098.7962997792742]]]
plot_polygon(polygon_list_)