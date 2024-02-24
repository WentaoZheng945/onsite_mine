# 内置库 
import math
import os
import sys
import copy
from itertools import combinations
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')) )

# 第三方库
import json
import numpy as np
from shapely.geometry import Polygon

# 自定义库
import common.utils as utils
from map_expansion.map_api import TgScenesMap
from map_expansion.bit_map import BitMap
from dynamic_scenes.lookup import CollisionLookup
from dynamic_scenes.observation import Observation
from dynamic_scenes.socket_module import Client




class ReplayInfo():
    """用于存储回放测试中用以控制背景车辆的所有数据
        背景车轨迹信息 vehicle_traj
        vehicle_traj = {
            "vehicle_id_0":{
                "shape":{
                    "vehicle_type":"PickupTruck",
                    "length":5.416,
                    "width":1.947,
                    "height":1.886,
                    "locationPoint2Head":2.708,
                    "locationPoint2Rear":2.708
                },
                "t_0":{
                    "x":0,
                    "y":0,
                    "v_mps":0,
                    "acc_mpss":0,
                    "yaw_rad":0,
                    "yawrate_radps":0,
                },
                "t_1":{...},
                ...
            },
            "vehicle_id_1":{...},
            ...
        }
        主车轨迹信息,只包含当前帧信息
        ego_info = {
            "shape":{
                        "vehicle_type":"MineTruck_NTE200",
                        "length":13.4,
                        "width":6.7,
                        "height":6.9,
                        "min_turn_radius":14.2,
                        "locationPoint2Head":2.708,
                        "locationPoint2Rear":2.708
                    },
            "x":0,
            "y":0,
            "v_mps":0,
            "acc_mpss":0,
            "yaw_rad":0,
            "yawrate_radps":0
        }
        地图相关信息,具体介绍地图解析工作的教程 markdown 文档待编写
        self.hdmaps = {
            'image_mask':bitmap的ndarray,
            'tgsc_map':语义地图
            }
        测试环境相关信息 test_setting
        test_setting = {
            "t":,
            "dt":,
            "max_t",
            "goal":{
                "x":[-1,-1,-1,-1],
                "y":[-1,-1,-1,-1]
            },
            "end":,
            "scenario_type":,
            "enter_loading_flag":,
            "enter_loading_time":,
            "scenario_name":,
            "map_type":,
            "start_ego_info" 
        }

    """


    def __init__(self):
        self.vehicle_traj = {}  # 背景车轨迹
        self.ego_info = {
            "shape":{
                        "vehicle_type":"MineTruck_NTE200",
                        "length":13.4,
                        "width":6.7,
                        "height":6.9,
                        "min_turn_radius":14.2,
                        "locationPoint2Head":2.708,
                        "locationPoint2Rear":2.708
                    },
            "x":0,
            "y":0,
            "v_mps":0,
            "acc_mpss":0,
            "yaw_rad":0,
            "yawrate_radps":0,
        }
        self.hdmaps = {}
        self.test_setting = {
            "t":0,
            "dt":0.1,
            "max_t":10,
            "goal":{
                "x":[1,2,3,4 ],
                "y":[1,2,3,4 ],
                "heading": None
            },# goal box:4 points [x1,x2,x3,x4],[y1,y2,y3,y4]
            "end":-1,
            "scenario_name":None,
            "scenario_type":None,
            "enter_loading_flag":False,
            "enter_loading_time":0.00,
            "x_min":None,
            "x_max":None,
            "y_min":None,
            "y_max":None,
            "start_ego_info":None  # 用不到，不会读取
        } #同 Observation.test_setting


    def _add_vehicle_shape(self, id: int, traj_info: dict = None):
        """ 
        该函数实现向vehicle_trajectiry中添加背景车轨迹信息的功能——增加车辆形状
        """
        if id not in self.vehicle_traj.keys():
            self.vehicle_traj[id] = {}
            self.vehicle_traj[id]['shape'] = {}
        self.vehicle_traj[id]['shape']= traj_info['VehicleShapeInfo']
     
            
    def _add_vehicle_traj(self,id: int, traj_info: dict = None):
        """ 
        该函数实现向vehicle_trajectiry中添加背景车轨迹信息的功能-增加车辆状态、轨迹
        """
        for index,_ in enumerate(traj_info['states']['x']):
            t = traj_info['StartTimeInScene']+ index * self.test_setting['dt']
            str_t = str(round(float(t),2))  # 注意这里的key为时间，是str
            if str_t not in self.vehicle_traj[id].keys(): 
                self.vehicle_traj[id][str_t]={}
            for key,value in zip(['x','y','yaw_rad','v_mps','yawrate_radps','acc_mpss'],
                                  [traj_info['states']['x'][index][0],traj_info['states']['y'][index][0],
                                   traj_info['states']['yaw_rad'][index][0],traj_info['states']['v_mps'][index][0],
                                   traj_info['states']['yawrate_radps'][index][0],traj_info['states']['acc_mpss'][index][0]]):
                if value is not None:
                    self.vehicle_traj[id][str_t][key] = (np.around(value,5))  # 轨迹保留5位小数
    
    
    def add_ego_info(self,ego_info):
        """
        该函数实现向test_setting中添加自车初始状态信息
        """
        self.ego_info['x'] = ego_info['states']['x']
        self.ego_info['y'] = ego_info['states']['y']
        self.ego_info['yaw_rad'] = ego_info['states']['yaw_rad']
        self.ego_info['v_mps'] = ego_info['states']['v_mps']
        self.ego_info['yawrate_radps'] = ego_info['states']['yawrate_radps']
        self.ego_info['acc_mpss'] = ego_info['states']['acc_mpss']
        self.ego_info['shape'] = ego_info['VehicleShapeInfo']
        
        
    def _add_settings(self,scenario_name=None,scenario_type=None):
        """
        该函数实现向test_setting中添加場景名稱和場景類型
        """
        for key,value in zip(['scenario_name','scenario_type'],
                              [scenario_name,scenario_type]):
            if value is not None:
                self.test_setting[key] = value


    def _init_vehicle_ego_info(self,one_scenario: dict = None):
        """
        该函数实现向ego_info中增加主车信息的功能
        注意:ego_info中只含有主车当前帧的信息
        """
        self.ego_info['shape'] = one_scenario['ego_info']['VehicleShapeInfo']
        self.ego_info['x'] = one_scenario['ego_info']['start_states']['x']
        self.ego_info['y'] = one_scenario['ego_info']['start_states']['y']
        self.ego_info['yaw_rad'] = one_scenario['ego_info']['start_states']['yaw_rad']
        self.ego_info['v_mps'] = one_scenario['ego_info']['start_states']['v_mps']
        self.ego_info['acc_mpss'] = one_scenario['ego_info']['start_states']['acc_mpss']
        self.ego_info['yawrate_radps'] = one_scenario['ego_info']['start_states']['yawrate_radps']


    def _get_dt_maxt(self,one_scenario: dict = None):
        """
        该函数实现得到最大仿真时长阈值以及采样率的功能,最大最小xy范围
        """
        self.test_setting['max_t']= one_scenario['max_t']
        self.test_setting['dt']= one_scenario['dt']
        self.test_setting['x_min']= one_scenario['x_min']
        self.test_setting['x_max']= one_scenario['x_max']
        self.test_setting['y_min']= one_scenario['y_min']
        self.test_setting['y_max']= one_scenario['y_max']
         



class ReplayParser():
    """
    解析场景文件
    """
    def __init__(self):
        self.replay_info = ReplayInfo()


    def parse(self,scenario:dict) -> ReplayInfo:
        """解析动态场景.动态场景包括: 
        1) 自车信息,self.replay_info.ego_info;
        2) 高精度地图信息,self.replay_info.hdmaps;
        3) 测试场景信息,self.replay_info.test_setting;
        4) 其它背景车辆全部轨迹信息,self.replay_info.vehicle_traj;
        
        Args:
            scenario (dict): 动态场景输入信息.

        Returns:
            ReplayInfo: 动态场景信息解析结果.
        """
        # 场景名称与测试类型
        self.replay_info._add_settings(scenario_name=scenario['data']['scene_name'],
                                      scenario_type=scenario['data']['scene_name'].split("_")[1])
        dir_scene_file = scenario['data']['dir_scene_file']
        self._parse_scenario(dir_scene_file) # 解析多车场景
        self._parse_hdmaps(scenario) # 解析 地图文件
        
        return self.replay_info


    def _parse_scenario(self, dir_scene_file:str):
        """解析多车场景.
        """
        with open(dir_scene_file,'r') as f:
            one_scenario = json.load(f)

        # 1) 获取ego车辆的目标区域,goal box
        self.replay_info.test_setting['goal'] = one_scenario['goal']
        self.replay_info.test_setting['goal'].setdefault('heading', None)
        
        # 2) 步长,最大时间,最大最小xy范围
        self.replay_info._get_dt_maxt(one_scenario)
        
        # 3) 读取ego车初始信息
        self.replay_info._init_vehicle_ego_info(one_scenario)

        # 读取背景车信息，包括：车辆形状信息以及轨迹信息
        for idex,value_traj_segment in enumerate(one_scenario['TrajSegmentInfo']):
            # if value_traj_segment['TrajSetToken'] != "ego":
            num_vehicle = idex +1
            # 4) 读取车辆长度与宽度等形状信息,录入replay_info.id从1开始
            self.replay_info._add_vehicle_shape(
                id=num_vehicle,
                traj_info=value_traj_segment)
            # 5) 以下读取背景车相关信息,车辆编号从1号开始,轨迹信息记录在vehicle_traj中
            self.replay_info._add_vehicle_traj(
                id=num_vehicle,
                traj_info=value_traj_segment)

        return self.replay_info
        
    
    def _parse_hdmaps(self,scenario:dict) -> None:
        """解析高清地图文件并更新到replay_info.
        Args: 这里的scenario为下一个待测场景列表中下一个待测场景对应的dict
        功能:
        1)获取可行驶区域的mask图信息.
        2)获取rgb图的目录(用于可视化) .
        3)加载路网结构.
        """
        # 初始化相关路径
        dataroot = scenario['file_info']['dir_maps']
        location = scenario['file_info']['location']
        
        # 1) 获取mask图信息并确定最大方形区域
        self._load_mask_and_calculate_square_region(dataroot,location)
        
        # 2) 获取rgb图信息 (  如果需要 )
        # self._load_rgb_image(dataroot,location)
        
        # 3) 加载路网信息
        self._load_road_network(dataroot,location)
        
        
    def _load_mask_and_calculate_square_region(self,dataroot:str,location:str) -> None:
        """加载mask并确定最大的方形区域.
            Args:
                dataroot: maps的根目录位置
                location: jiangxi_jiangtong/guangdong_dapai
        """
        self.replay_info.hdmaps['image_mask'] = BitMap(dataroot,location,'bitmap_mask')  # 得到整个二进制图的ndarray
        
        # 整合所有的坐标点
        x_coords = (
            [self.replay_info.ego_info['x']] + 
            self.replay_info.test_setting['goal']['x'] +
            [self.replay_info.test_setting['x_min'],self.replay_info.test_setting['x_max']]
        )  # 包括主车初始横坐标、目标区域横坐标、以及整张地图横坐标
        y_coords = (
            [self.replay_info.ego_info['y']] + 
            self.replay_info.test_setting['goal']['y'] +
            [self.replay_info.test_setting['y_min'],self.replay_info.test_setting['y_max']]
        )
        
        # 根据坐标确定最大的方形框
        x_min,x_max = min(x_coords),max(x_coords)
        y_min,y_max = min(y_coords),max(y_coords)
        max_range = max(x_max - x_min,y_max - y_min)  # 确定正方形边长
        x_center,y_center = (x_min + x_max) / 2,(y_min + y_max) / 2  # 确定正方形中心
        utm_local_range = (
            x_center - max_range / 2,y_center - max_range / 2,
            x_center + max_range / 2,y_center + max_range / 2
        )  # 确定坐标范围
        self.replay_info.hdmaps['image_mask'].load_bitmap_using_utm_local_range(utm_local_range,20,20)  # 希望加载的位图范围能比坐标范围大一点
        # 局部位图二进制范围结果储存在self.replay_info.hdmaps['image_mask'].image_ndarray_local中
        
    
    def _load_rgb_image(self,dataroot:str,location:str) -> None:
        """加载地图的RGB图像.
        """
        self.replay_info.hdmaps['image_rgb'] = BitMap(dataroot,location,'bitmap_rgb',is_transform_gray=True)
    
    
    def _load_road_network(self,dataroot:str,location:str) -> None:
        """加载地图的路网信息.
        """
        self.replay_info.hdmaps['tgsc_map'] = TgScenesMap(dataroot,location)  #语义地图信息
        
          
        
        

class ReplayController():
    def __init__(self, kinetics_mode='complex'):
        self.control_info = ReplayInfo()
        self.kinetics_mode = None
        if kinetics_mode == 'complex':
            self.kinetics_mode = kinetics_mode
            self.client = Client()


    def init(self,control_info:ReplayInfo,collision_lookup:CollisionLookup) -> Observation:
        self.control_info = control_info
        self.collision_lookup = collision_lookup
        return self._get_initial_observation()

    def step(self,action,old_observation:Observation,collision_lookup:CollisionLookup) -> Observation:
        action = self._action_cheaker(action)
        if self.kinetics_mode == 'complex':
            new_observation = self._update_ego_and_t_kinetics(action,old_observation)
        else:
            new_observation = self._update_ego_and_t(action, old_observation)
        new_observation = self._update_other_vehicles_to_t(new_observation)
        new_observation = self._update_end_status(new_observation)
        if self.kinetics_mode == 'complex':
            if new_observation.test_setting['end'] != -1:
                self.client.client_send_sock.close()
                self.client.client_receive_sock.close()
        return new_observation


    def _action_cheaker(self,action):
        a = np.clip(action[0],-15,15)
        rad = np.clip(action[1],-1,1)
        gear = action[2]
        if gear not in [1, 2, 3]:
            raise ValueError("不支持{0}档位，请选择合适档位".format(gear))
        return (a,rad,gear)


    def _get_initial_observation(self) -> Observation:
        observation = Observation()
        # vehicle_info
        observation.vehicle_info["ego"] = self.control_info.ego_info  # 初始化主车信息
        observation = self._update_other_vehicles_to_t(observation)  # 初始化背景车信息
        # hdmaps info
        observation.hdmaps = self.control_info.hdmaps
        # test_setting
        observation.test_setting = self.control_info.test_setting
        observation = self._update_end_status(observation)
    
        return observation


    def _update_ego_and_t(self,action:tuple,old_observation:Observation) -> Observation:
        # 拷贝一份旧观察值
        new_observation = copy.copy(old_observation)
        # 首先修改时间,新时间=t+dt
        new_observation.test_setting['t'] = float(
            old_observation.test_setting['t'] +
            old_observation.test_setting['dt']
        )
        # 修改本车的位置,方式是前向欧拉更新,1.根据旧速度更新位置;2.然后更新速度.
        # 速度和位置的更新基于自行车模型.
        # 首先分别取出加速度和方向盘转角
        a,rot,gear = action
        # 取出步长
        dt = old_observation.test_setting['dt']
        # 取出本车的各类信息
        x,y,v,yaw,= [float(old_observation.vehicle_info['ego'][key]) for key in ['x','y','v_mps','yaw_rad']]
        width,length  = old_observation.vehicle_info['ego']['shape']['width'],old_observation.vehicle_info['ego']['shape']['length']

        # 首先根据旧速度更新本车位置
        new_observation.vehicle_info['ego']['x'] = x + \
                                                   v * np.cos(yaw) * dt  # 更新X坐标

        new_observation.vehicle_info['ego']['y'] = y + \
                                                   v * np.sin(yaw) * dt  # 更新y坐标

        new_observation.vehicle_info['ego']['yaw_rad'] = yaw + \
                                                     v / length * 1.7 * np.tan(rot) * dt  # 更新偏航角

        new_observation.vehicle_info['ego']['v_mps'] = v + a * dt  # 更新速度
        if new_observation.vehicle_info['ego']['v_mps'] < -10:
            new_observation.vehicle_info['ego']['v_mps'] = -10

        new_observation.vehicle_info['ego']['acc_mpss'] = a  # 更新加速度
        return new_observation

    def _judge_gear(self, a, rot):
        # TODO 完成人字形节点判断，进而判断档位
        return 1

    def _update_ego_and_t_kinetics(self,action:tuple,old_observation:Observation) -> Observation:
        # 拷贝一份旧观察值
        new_observation = copy.copy(old_observation)
        # 首先修改时间,新时间=t+dt
        new_observation.test_setting['t'] = float(
            old_observation.test_setting['t'] +
            old_observation.test_setting['dt']
        )
        # 修改本车的位置,方式是前向欧拉更新,1.根据旧速度更新位置;2.然后更新速度.
        # 速度和位置的更新基于自行车模型.
        # 首先分别取出加速度和方向盘转角
        a,rot,gear = action
        rot = rot * (180 / math.pi)
        # 取出步长
        dt = old_observation.test_setting['dt']

        # 判断档位
        # gear = self._judge_gear(a, rot)
        unpacked_data = self.client.send_and_receive(gear, a, rot)

        # 更新本车位置
        new_observation.vehicle_info['ego']['x'] = unpacked_data[1]
        new_observation.vehicle_info['ego']['y'] = unpacked_data[2]
        # 更新航向角
        new_observation.vehicle_info['ego']['yaw_rad'] = unpacked_data[0] * (math.pi / 180)
        # 更新速度
        new_observation.vehicle_info['ego']['v_mps'] = unpacked_data[3]
        if new_observation.vehicle_info['ego']['v_mps'] < -10:
            new_observation.vehicle_info['ego']['v_mps'] = -10
        # 更新加速度
        new_observation.vehicle_info['ego']['acc_mpss'] = a
        return new_observation


    def _update_other_vehicles_to_t(self,old_observation:Observation) -> Observation:
        """已知仿真时刻，把背景车信息更新至对应时刻"""
        # 删除除了ego之外的车辆观察值
        new_observation = copy.copy(old_observation)  # 复制一份旧观察值
        new_observation.vehicle_info = {}
        # 将本车信息添加回来
        new_observation.vehicle_info['ego'] = old_observation.vehicle_info['ego']
        # 根据时间t,查询control_info,赋予新值
        t = old_observation.test_setting['t']
        t = str(np.around(t,3))  # t保留3位小数,与生成control_info时相吻合
        for vehi in self.control_info.vehicle_traj.items():
            id = vehi[0]  # 车辆id
            info = vehi[1]  # 车辆的轨迹信息
            if t in info.keys():
                new_observation.vehicle_info[id] = {}
                for key in ['x','y','yaw_rad','v_mps','yawrate_radps','acc_mpss']:
                    new_observation.vehicle_info[id][key] = info[t][key]
                    # if key == 'acc_mpss' or key == 'v_mps' :
                    #     new_observation.vehicle_info[id][key] = 2.0*info[t][key] #######
                        
                new_observation.vehicle_info[id]['shape'] = info['shape']
        return new_observation


    def _update_end_status(self,observation:Observation) -> Observation:
        """计算T时刻,测试是否终止,更新observation.test_setting中的end值
            end=
                -1:回放测试正常进行; 
                1:回放测试运行完毕;
                2:ego车与其它车辆发生碰撞;
                3:ego车与道路边界发生碰撞(驶出道路边界);
                4:ego车到达目标区域               
        """
        status_list = [-1]

        # 检查主车与背景车是否发生碰撞
        if self._collision_detect(observation):
            status_list += [2]  # 添加状态
            print("###log### 主车与背景车发生碰撞\n")
            

        # 检查是否已到达场景终止时间max_t
        if observation.test_setting['t'] >= self.control_info.test_setting['max_t']:
            status_list += [1]
            print("###log### 已到达场景终止时间max_t\n")
            
    
        # 检查是否与道路边界碰撞
        local_x_range = observation.hdmaps['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_x_range']
        local_y_range = observation.hdmaps['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_y_range']
        collision_flag = self.collision_lookup.collisionDetection(observation.vehicle_info['ego']['x']-local_x_range[0],
                                                        observation.vehicle_info['ego']['y']-local_y_range[0],
                                                        observation.vehicle_info['ego']['yaw_rad'],
                                                        observation.hdmaps['image_mask'].image_ndarray)
        if collision_flag == True:
            status_list += [3]
            print("###log### 主车与道路边界碰撞\n")
            
        # check target area
        if observation.test_setting["scenario_type"] == "intersection" or observation.test_setting["scenario_type"] == "unloading":
            if utils.is_inside_polygon(observation.vehicle_info['ego']['x'],observation.vehicle_info['ego']['y'],observation.test_setting['goal']):
                status_list += [4]
                print("###log### 主车已到达目标区域\n")
        elif observation.test_setting["scenario_type"] == "loading":
            if observation.test_setting["enter_loading_flag"] == False:
                if observation.vehicle_info['ego']['v_mps'] == 0 and \
                        utils.is_inside_polygon(observation.vehicle_info['ego']['x'], observation.vehicle_info['ego']['y'], observation.test_setting['goal']):
                    # status_list += [4]
                    print("###log### 主车已驶入装载区域，正在进行最终位姿调整\n")
                    observation.test_setting["enter_loading_flag"] = True
                    observation.test_setting['enter_loading_time'] = observation.test_setting['t']
            elif observation.test_setting["enter_loading_flag"] == True:
                adjusting_time = float(observation.test_setting['t'] - observation.test_setting["enter_loading_time"])  # 注意这里需要检查数据类型
                if adjusting_time > 5:
                    status_list += [4]
                    print("###log### 主车已驶入装载区域，且到达位姿调整时间上限\n")


        # 从所有status中取最大的那个作为end.
        observation.test_setting['end'] = max(status_list)
        return observation


    def _collision_detect(self,observation:Observation) -> bool:
        poly_zip = []
        self.vehicle_index = []  # 这里为了判断哪两辆车发生了碰撞,定义了列表用来存放车辆名称,其index与poly_zip中车辆的图形索引相对应
        # 当测试时间大于0.5秒时,遍历所有车辆,绘制对应的多边形.
        if observation.test_setting['t'] > 0.5:
            for index,vehi in observation.vehicle_info.items():
                self.vehicle_index += [index]
                poly_zip += [self._get_poly(vehi)]

        # 检测主车是否与背景车碰撞
        for a,b in combinations(poly_zip,2):
            if self.vehicle_index[poly_zip.index(a)] == 'ego' or self.vehicle_index[poly_zip.index(b)] == 'ego':
                if a.intersects(b):
                    return True
                else:
                    continue
        return False


    def _get_poly(self,vehicle:dict) -> Polygon:
        """根据车辆信息,通过shapely库绘制矩形.这是为了方便地使用shapely库判断场景中的车辆是否发生碰撞
        """
        # 提取车辆shape中的属性
        length = vehicle['shape']['length']
        width = vehicle['shape']['width']
        locationPoint2Head = vehicle['shape']['locationPoint2Head']
        locationPoint2Rear = vehicle['shape']['locationPoint2Rear']
    
        front_left_corner,front_right_corner,rear_left_corner,rear_right_corner = utils.calculate_vehicle_corners(
            length,width,locationPoint2Head,locationPoint2Rear,vehicle['x'],vehicle['y'],vehicle['yaw_rad'])

        # 通过车辆矩形的4个顶点,可以绘制出对应的长方形
        poly = Polygon([
            (front_left_corner[0],front_left_corner[1]),
            (front_right_corner[0],front_right_corner[1]),
            (rear_right_corner[0],rear_right_corner[1]),
            (rear_left_corner[0],rear_left_corner[1]),
            (front_left_corner[0],front_left_corner[1])
        ]).convex_hull
        return poly




class Controller():
    """控制车辆运行
    """
    def __init__(self) -> None:
        self.observation = Observation()
        self.parser = None
        self.control_info = None
        self.controller = None
        self.mode = 'replay'


    def init(self,scenario:dict,collision_lookup:CollisionLookup, kinetics_mode:str) -> Tuple:
        """初始化运行场景,给定初始时刻的观察值

        Parameters
        ----------
        input_dir :str
            测试输入文件所在位置
                回放测试:包含ScenariosResultes、TrajDataResultes、other目录,存放场景信息,轨迹片段集合、车辆配置信息等
                交互测试:
        mode :str
            指定测试模式
                回放测试:replay
                交互测试:interact
        Returns
        -------
        observation :Observation
            初始时刻的观察值信息,以Observation类的对象返回.
        """
        self.mode = scenario['test_settings']['mode']
        if self.mode == 'replay':
            self.parser = ReplayParser()
            self.controller = ReplayController(kinetics_mode)
            self.control_info = self.parser.parse(scenario)
            self.observation = self.controller.init(self.control_info,collision_lookup)
            self.traj = self.control_info.vehicle_traj
        return self.observation,self.traj


    def step(self,action,collision_lookup:CollisionLookup):
        self.observation = self.controller.step(action,self.observation,collision_lookup)
        return self.observation




if __name__ == "__main__":
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '/home/czf/project_czf/20231010_onsite_mine/devkit/inputs'))
    scenes_file = os.path.join(input_dir,'ScenariosResultes','Scenario-jiangtong_intersection_1_1_2.json')

    with open(scenes_file,'r') as f:
        jsondata = json.load(f)
            
    pass
