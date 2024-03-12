# 导入内置库
import sys
import os
import time
import os
import json
import importlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from pathlib import Path as PathlibPath

# 导入第三方库
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from typing import Dict,List,Tuple,Optional,Union,Any

# # 添加必要的路径到sys.path
# 导入onsite-mine相关模块（库中含有-,因此使用动态importlib动态导入包）
## onsite-mine.dynamic_scenes模块
scenarioOrganizer = importlib.import_module("onsite-mine.dynamic_scenes.scenarioOrganizer1")
env = importlib.import_module("onsite-mine.dynamic_scenes.env")

# 导入本地模块
from predictor import Predictor
from planner import Planner
from simple_control import MotionController

# 检查对应文件夹是否存在
def check_dir(target_dir):
    """check path"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)

dir_current_file = os.path.dirname(__file__)  # 'Algorithm_demo_1\planner'
dir_parent_1 = os.path.dirname(dir_current_file) # 'Algorithm_demo_1'


if __name__ == "__main__":
    dir_inputs = os.path.abspath(os.path.join(dir_parent_1, 'inputs'))  # 场景文件的位置
    dir_outputs = os.path.abspath(os.path.join(dir_parent_1, 'outputs'))  # 输出轨迹的位置
    dir_save_img = os.path.abspath(os.path.join(dir_parent_1, 'onsite_images_saved'))  # 图像保存位置
    tic = time.time()
    so = scenarioOrganizer.ScenarioOrganizer()  # 初始化场景管理模块，用于加载测试场景
    envi = env.Env()  # 初始化测试环境
    # 根据配置文件config.py装载场景,指定输入文件夹即可,会自动检索配置文件
    so.load(dir_inputs, dir_outputs)  # 根据配置文件config.py加载待测场景，指定输入文件夹即可，会自动检索配置文件
    formatted_so_config = json.dumps(so.config, indent=4, ensure_ascii=False)  # 清晰展示各種設置，没有特殊含义
    print(f"###log### <测试参数>\n{formatted_so_config}\n")

    while True:
        scenario_to_test = so.next()  # !使用场景管理模块给出下一个待测场景
        if scenario_to_test is None:
            break  # !如果场景管理模块给出None,意味着所有场景已测试完毕.
        print(f"###log### <scene-{scenario_to_test['data']['scene_name']}>\n")  # 输出待测场景信息
        try:
            # 使用env.make方法初始化当前测试场景
            observation, traj, client = envi.make(scenario=scenario_to_test, save_img_path=dir_save_img,
                                                  kinetics_mode='simple')

            ########### 算法 部分1:初始化 ###########
            # TODO 交叉路口单车通行任务:构建整个场景的预测器,预测所有的车辆;构建规划器\控制器,控制ego车
            predictor = Predictor(time_horizon=5.0)
            planner = Planner(observation)
            path_planned, spd_planned = planner.process(observation)
            lat_controller = MotionController()
            step_sum = int(observation["test_setting"]['max_t'] / observation["test_setting"]['dt'])
            ########### 算法 部分1:初始化 ###########

            # 逐帧进行仿真，触发仿真停止条件时结束
            # 当测试还未进行完毕,即观察值中test_setting['end']还是-1的时候
            while observation['test_setting']['end'] == -1:
                ########### 算法 部分2:执行 ###########
                traj_future, _ = predictor.predict(observation, traj, predictor_type="CACV_PREDICTOR")
                # traj_future,_ = predictor.predict(observation,traj,predictor_type ="LANE_SEQUENCE_PREDICTOR")
                action = lat_controller.process(observation['vehicle_info'], path_planned, traj_future, observation)
                # action = (3,-0.55)
                action = action + (1,)  # 添加档位信息，默认为前进档位
                ########### 算法 部分2:执行 ###########
                observation = envi.step(action, traj_future, observation, traj)  # 根据车辆的action,更新场景,并返回新的观测值.

        except Exception as e:
            print(repr(e))
            if client is not None:
                client.close_sockets()
        finally:
            # 如果测试完毕，将测试结果传回场景管理模块（ScenarioOrganizer)
            so.add_result(scenario_to_test, observation['test_setting']['end'])
            # 在每一次测试最后都关闭可视化界面，避免同时存在多个可视化
            plt.close()

    toc = time.time()
    print("###log### 总用时:", toc - tic, "秒\n")
         


