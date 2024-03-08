# 内置库 
import os
import sys
import shlex
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# 第三方库
from typing import Dict,List,Tuple,Optional,Union

# 自定义库
from dynamic_scenes.observation import Observation
from dynamic_scenes.controller import Controller
from dynamic_scenes.recorder import Recorder
from dynamic_scenes.visualizer import Visualizer
from dynamic_scenes.lookup import CollisionLookup
from dynamic_scenes.kinetics_model import KineticsModelStarter




class Env():
    """仿真环境读取及迭代过程,simulation"""
    def __init__(self):
        self.controller = Controller()
        self.recorder = Recorder()
        self.visualizer = Visualizer()


    def make(self,scenario:dict,collision_lookup:CollisionLookup,read_only=False, save_img_path='',kinetics_mode='simple') -> Tuple:
        """第一次进入读取环境信息.

        Args:
            scenario (dict): 动态场景输入信息.
            collision_lookup (CollisionLookup): 用于 ego车与(栅格化mask)边界 进行碰撞检测的预置数据.
            read_only (bool, optional): _description_. Defaults to False.

        Returns:
            Observation: 当前时刻环境观察结果;
            traj:全局的背景车辆轨迹数据;
        """
        observation,traj = self.controller.init(scenario,collision_lookup,kinetics_mode)
        if kinetics_mode == "complex":
            Starter = KineticsModelStarter(observation)
            self.client = Starter.get_client
        elif kinetics_mode == "simple":
            self.client = None
        else:
            raise ValueError("暂不提供这种动力学模式，请在simple和complex中选择模式！")
        self.recorder.init(observation,scenario['file_info']['dir_outputs'],read_only)
        self.visualizer.init(observation,
                             scenario['test_settings']['visualize'],
                             scenario['test_settings']['save_fig_whitout_show'],
                             img_save_path=save_img_path) # 此处通过查看配置参数,True,设置运行过程中可视化打开;
        
        return observation.format(),traj,self.client


    def step(self,action:Tuple[float,float,int],traj_future:Dict,observation_last:Observation,traj:Dict,collision_lookup:CollisionLookup) -> Observation:
        """迭代过程"""
        observation = self.controller.step(action,collision_lookup,self.client)  # 使用车辆运动学模型单步更新场景;
        self.recorder.record(observation)
        # self.visualizer.update(observation)  
        self.visualizer.update(observation,traj_future,observation_last,traj)# 更新场景后,使用更新的ego车辆位置进行可视化;【CZF】添加预测轨迹+真实轨迹的比较
        
        return observation.format()





if __name__ == "__main__":
    import time
    demo_input_dir = r"demo/demo_inputs"
    demo_ouput_dir = r"demo/demo_outputs"
    tic = time.time()
    env = Env()

    from dynamic_scenes.scenarioOrganizer import ScenarioOrganizer
    # 实例化场景管理模块(ScenairoOrganizer)和场景测试模块(Env)
    so = ScenarioOrganizer()
    # 根据配置文件config.py装载场景,指定输入文件夹即可,会自动检索配置文件
    so.load(demo_input_dir,demo_ouput_dir)
    num_scenario = len(so.scenario_list)
    for i in range(num_scenario):
        scenario_to_test = so.next()
        print(scenario_to_test)
        observation = env.make(scenario_to_test,demo_ouput_dir,visilize=True)
        while observation.test_setting['end'] == -1:
            observation = env.step([-1,0])
            # print(observation.vehicle_info['ego'])
    toc = time.time()
    print(toc - tic)
