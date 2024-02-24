 # 内置库 
import math
import statistics

# 第三方库
import numpy as np
from shapely.geometry import Point, Polygon
from typing import Dict,List,Tuple,Optional,Union



# class Point2D:
#     """2D平面的位姿点."""
#     def __init__(self,x,y,yaw_rad:float=0.0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw_rad


class Planner:
    """ego车的轨迹规划器.
    注:业界一般做法是 path planning + speed planning .
    """
    def __init__(self,observation):
        self._goal_x = statistics.mean(observation['test_setting']['goal']['x'])
        self._goal_y = statistics.mean(observation['test_setting']['goal']['y'])
        self._observation = observation
    
    
    def process(self,observation):
        """规划器主函数.
        注：该函数功能设计举例
        1）进行实时轨迹规划;
        2) 路径、速度解耦方案:给出局部轨迹规划器的实时求解结果--待行驶路径、待行驶速度；
        
        输入:observation——环境信息;
        输出: 路径、速度解耦方案:给出局部轨迹规划器的实时求解--
            待行驶路径（离散点序列）、待行驶速度（与路径点序列对应的速度离散点序列）
        """
        path_planned = self.get_ego_reference_path_to_goal(observation)
        spd_planned = None
        
        return path_planned,spd_planned
        
        
    def get_ego_reference_path_to_goal(self,observation):
        """获取HD Map中ego车到达目标区域的参考路径.
        注1：该参考路径仅表示了对于道路通行规则的指示。
            数据处理来源:1)在地图中手动标记dubinspose;2)生成dubins curve;3)离散拼接.
        注2:显然该参考路径存在以下缺陷--
            1)在实际道路中使用该参考路径跟车行驶时不符合曲率约束;2)onsite_mine仿真中存在与边界发生碰撞的风险.
        注3:onsite_mine自动驾驶算法设计鼓励参与者设计符合道路场景及被控车辆特性的实时轨迹规划算法.
        
        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径(拼接后).
        """
        if 0: #仅仅适用于 会车场景 jiangtong-9-1-1
            ego_connected_path_tokens = ["path-107","path-93"]
            return self.get_connected_waypoints_from_multi_path(observation,ego_connected_path_tokens)
        #################更新参数#################
        self._ego_x = observation['vehicle_info']['ego']['x']
        self._ego_y = observation['vehicle_info']['ego']['y']
        self._ego_v = observation['vehicle_info']['ego']['v_mps']
        self._ego_yaw = observation['vehicle_info']['ego']['yaw_rad']
        
        #################定位主车和目标点所在几何#################
        ego_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._ego_x,self._ego_y)
        ego_polygon_id = int( ego_polygon_token.split('-')[1])
        # goal_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._goal_x,self._goal_y)
        
        #################获取目标车最匹配的dubinspose(在多边形所属的link_dubinspose_tokens中查找)#################
        ego_dubinspose_token = observation['hdmaps_info']['tgsc_map'].get_dubinspose_token_from_polygon\
                            ((self._ego_x,self._ego_y,self._ego_yaw),ego_polygon_token)
        # ego_dubinspose_id = int( ego_dubinspose_token.split('-')[1])
        
        #################搜索go_polygon_token 到goal_polygon_token的参考路径可能 #################
        # ego_dubinspose_token 作为 起点、终点的path拿到
        link_referencepath_tokens_ego_polygon = observation['hdmaps_info']['tgsc_map'].polygon[ego_polygon_id]['link_referencepath_tokens']
        
        # 去除掉 不包含 ego_dubinspose_token 的 path
        temp_ego_path_tokens=[]
        for _,path_token in enumerate(link_referencepath_tokens_ego_polygon):
            path_id = int( path_token.split('-')[1])
            link_dubinspose_tokens = observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['link_dubinspose_tokens']
            if ego_dubinspose_token not in link_dubinspose_tokens:
                pass
            else:
                temp_ego_path_tokens.append( path_token )
                if observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['type'] == 'connector_path':
                    pass
                else:
                    only_one_path_token = path_token 
                    only_one_path_id= path_id 
                    
                    
        
        # 广度优先搜索,一般搜索不超过3层
        path_connect_tree ={"layer_1":only_one_path_token, "layer_2":{} }
        second_layer_path_tokens =  observation['hdmaps_info']['tgsc_map'].reference_path[only_one_path_id]['outgoing_tokens']
        
        # 第二层
        for _,token_2 in enumerate(second_layer_path_tokens):#path_token_2
            id_2= int( token_2.split('-')[1]) #path_id_2
            if token_2 not in path_connect_tree["layer_2"]:
                path_connect_tree["layer_2"][token_2]={"flag_inside_goal_area":False,
                                                            "layer_3":{} }                
            ref_path = np.array( observation['hdmaps_info']['tgsc_map'].reference_path[id_2]['waypoints'])
            # 搜索终止条件:参考路径的waypoints有点waypoint在目标区域内部
            flag_inside_goal_area =self.has_waypoint_inside_goal_area(ref_path,
                                               observation['test_setting']['goal']['x'],
                                               observation['test_setting']['goal']['y'])
            if flag_inside_goal_area:
                path_connect_tree['layer_2'][token_2]["flag_inside_goal_area"]=True
                ego_connected_path_tokens =[path_connect_tree['layer_1'],
                                                token_2 ]
                return self.get_connected_waypoints_from_multi_path(observation,ego_connected_path_tokens)
            else:
                path_connect_tree['layer_2'][token_2]["flag_inside_goal_area"]=False
                outgoing_tokens = observation['hdmaps_info']['tgsc_map'].reference_path[id_2]['outgoing_tokens']
                for _,token in enumerate(outgoing_tokens):
                    if token not in path_connect_tree['layer_2'][token_2]["layer_3"]:
                        path_connect_tree['layer_2'][token_2]["layer_3"][token]={}        
        
        # 第三层        
        for _,token_2 in enumerate(second_layer_path_tokens): 
            thrid_layer_path_tokens = path_connect_tree['layer_2'][token_2]['layer_3']
            for _,token_3 in enumerate(thrid_layer_path_tokens): 
                id_3= int( token_3.split('-')[1])
                if not path_connect_tree['layer_2'][token_2]["layer_3"][token_3]:#空
                   path_connect_tree['layer_2'][token_2]["layer_3"][token_3]={"flag_inside_goal_area":False,
                                                                              "layer_4":{} }                
                ref_path = np.array( observation['hdmaps_info']['tgsc_map'].reference_path[id_3]['waypoints'])
                flag_inside_goal_area=self.has_waypoint_inside_goal_area(ref_path,
                                                observation['test_setting']['goal']['x'],
                                                observation['test_setting']['goal']['y'])
                if flag_inside_goal_area:
                    path_connect_tree['layer_2'][token_2]['layer_3'][token_3]['flag_inside_goal_area']=True
                    ego_connected_path_tokens =[path_connect_tree['layer_1'],
                                                token_2,
                                                token_3  ]
                    return self.get_connected_waypoints_from_multi_path(observation,ego_connected_path_tokens)
                    
                else:
                    path_connect_tree['layer_2'][token_2]['layer_3'][token_3]['flag_inside_goal_area']=False
                    outgoing_tokens = observation['hdmaps_info']['tgsc_map'].reference_path[id_3]['outgoing_tokens']
                    for _,token in enumerate(outgoing_tokens):
                        if token not in path_connect_tree['layer_2'][token_2]["layer_3"][token_3]['layer_4']:
                            # path_connect_tree['layer_2']['path-7']['layer_3']['path-70']['layer_4']
                            path_connect_tree['layer_2'][token_2]["layer_3"][token_3]['layer_4'][token]={}          
       
    
    @staticmethod       
    def has_waypoint_inside_goal_area(ref_path_waypoints:np.array=None,
                                      goal_area_x:List=None,
                                      goal_area_y:List=None,) ->bool:
        """计算参考路径的waypoints 是否 有点waypoint在目标区域内部.

        Args:
            ref_path_waypoints (np.array, optional): 参考路径的waypoints. Defaults to None.
            goal_area_x (List, optional): 目标区域x坐标列表. Defaults to None.
            goal_area_y (List, optional): 目标区域y坐标列表. Defaults to None.

        Returns:
            bool: 参考路径的waypoints 是否 有点waypoint在目标区域内部
        """
        if ref_path_waypoints is None or goal_area_x is None or goal_area_y is None:
            return False

        # Create Polygon object representing the goal area
        goal_area_coords = list(zip(goal_area_x, goal_area_y))
        goal_area_polygon = Polygon(goal_area_coords)

        # Check each waypoint
        for waypoint in ref_path_waypoints:
            x, y = waypoint[0], waypoint[1] 
            if goal_area_polygon.contains(Point(x, y)):
                return True
        return False
    
    
    def get_connected_waypoints_from_multi_path(self,observation,connected_path_tokens:List=None ):
        """ 获得多条路径拼接后的waypoints.
            waypoints(x,y,yaw,heigh,slope).

        Args:
            observation (_type_): 环境信息.
            connected_path_tokens (List, optional): _description_. Defaults to None.

        Returns:
            List: 多条路径拼接后的waypoints(x,y)
        """
        
        connected_waypoints = []
        
        # 方法1: 使用 extend() 方法
        # for token_path in connected_path_tokens:
        #     id_path = int(token_path.split('-')[1])
        #     temp_waypoints = observation['hdmaps_info']['tgsc_map'].reference_path[id_path]['waypoints']
        #     connected_waypoints.extend(temp_waypoints)
        
        # 方法2:使用列表推导式:大列表更快,直接在 C 语言级别实现的,不需要通过 Python 解释器.
        connected_waypoints = [point for token_path in connected_path_tokens for point in observation['hdmaps_info']['tgsc_map'].reference_path[int(token_path.split('-')[1])]['waypoints'] ]
        # 增加可读性
        # connected_waypoints = [
        #     point
        #     for token_path in connected_path_tokens
        #     for point in (
        #         observation['hdmaps_info']['tgsc_map'].reference_path[
        #             int(token_path.split('-')[1])
        #         ]['waypoints']
        #     )
        # ]

        return connected_waypoints
    
    
    def get_connected_waypoints_from_multi_path_array(self,observation,connected_path_tokens:List=None ):
        """获得多条路径拼接后的waypoints.使用np.array

        Args:
            connected_path_tokens (List, optional): _description_. Defaults to None.
        """
        # Initialize connected_waypoints as None
        connected_waypoints = None
        for token_path in connected_path_tokens:
            id_path = int(token_path.split('-')[1])
            
            # Get temp_waypoints from observation
            temp_waypoints = np.array(observation['hdmaps_info']['tgsc_map'].reference_path[id_path]['waypoints'])
            
            # Check if connected_waypoints is None, if so assign temp_waypoints to it
            # otherwise concatenate temp_waypoints to connected_waypoints
            if connected_waypoints is None:
                connected_waypoints = temp_waypoints
            else:
                connected_waypoints = np.concatenate((connected_waypoints, temp_waypoints), axis=0)
            
        return connected_waypoints
        
        
         
              
