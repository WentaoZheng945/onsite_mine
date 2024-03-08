# 内置库
import math
from queue import PriorityQueue
import statistics

# 第三方库
import numpy as np
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional, Union


class Planner:
    """
    ego车的规划器.
    包括：全局路径规划器；局部轨迹规划器；
    注:局部轨迹规划器业界一般做法是 path planning + speed planning .
    """

    def __init__(self, observation):
        self._goal_x = statistics.mean(observation["test_setting"]["goal"]["x"])
        self._goal_y = statistics.mean(observation["test_setting"]["goal"]["y"])
        self._observation = observation

    def process(self, observation):
        """规划器主函数.
        注：该函数功能设计举例
        0) 全局路径寻优,route.
        1）进行实时轨迹规划;
        2) 路径、速度解耦方案:给出局部轨迹规划器的实时求解结果--待行驶路径、待行驶速度；

        输入:observation——环境信息;
        输出: 路径、速度解耦方案:给出局部轨迹规划器的实时求解--
            待行驶路径（离散点序列）、待行驶速度（与路径点序列对应的速度离散点序列）
        """
        # 全局路径寻优
        route_path = self.get_ego_reference_path_to_goal(observation)
        path_planned = route_path
        spd_planned = None

        return path_planned, spd_planned

    def get_best_matching_path_token_from_polygon(self, veh_pose: Tuple[float, float, float], polygon_token: str, observation) -> str:
        """根据veh_pose(x,y,yaw)车辆定位位姿,从polygon_token所属的 link_referencepath_tokens 中匹配最佳参考路径.

        方法：
        1) 匹配2条path最近点;
        2）获取最佳path;

        Args:
            veh_pose (Tuple[float,float,float]):车辆的位姿.
            polygon_token (str):指定的polygon token.

        Returns:
            str:最佳匹配的 path_token,id_path.
        """
        semantic_map = observation["hdmaps_info"]["tgsc_map"]
        if not polygon_token.startswith("polygon-"):
            raise ValueError(f"Invalid polygon_token:{polygon_token}")

        id_polygon = int(polygon_token.split("-")[1])
        if id_polygon > len(semantic_map.polygon):
            raise IndexError(f"Polygon ID {id_polygon} out of bounds.请检查.")
        if semantic_map.polygon[id_polygon]["type"] == "intersection":
            raise IndexError(f"##log## Polygon ID = {id_polygon},目前未处理自车初始位置在交叉口区域的寻路逻辑.")

        link_referencepath_tokens = semantic_map.polygon[id_polygon]["link_referencepath_tokens"]

        candidate_paths = PriorityQueue()
        for _, path_token in enumerate(link_referencepath_tokens):
            id_path = int(path_token.split("-")[1])
            if semantic_map.reference_path[id_path]["type"] == "base_path":
                # 匹配最近点
                waypoint = self.find_nearest_waypoint(
                    waypoints=np.array(semantic_map.reference_path[id_path]["waypoints"]), veh_pose=veh_pose, downsampling_rate=5
                )
                yaw_diff = self.calc_yaw_diff_two_waypoints(waypoint1=(waypoint[0], waypoint[1], waypoint[2]), waypoint2=veh_pose)
                path_info = {"path_token": path_token, "id_path": id_path, "waypoint": waypoint, "yaw_diff": abs(yaw_diff)}
                candidate_paths.put((path_info["yaw_diff"], path_info))  # yaw_diff由小到大排序

        if candidate_paths.empty():
            raise ValueError(f"##log## Polygon ID = {id_polygon},所属路径均为connector_path,有问题.")
        # 得到同向最佳path的 token,id
        best_path_info = candidate_paths.get()  # 自动返回优先级最高的元素（优先级数值最小的元素）并从队列中移除它。

        return best_path_info[1]["path_token"], best_path_info[1]["id_path"]

    def find_nearest_waypoint(self, waypoints: np.array, downsampling_rate: int = 5, veh_pose: Tuple[float, float, float] = None):
        waypoints_downsampling = np.array(waypoints[::downsampling_rate])  # downsampling_rate,每5个路径点抽取一个点
        distances = np.sqrt((waypoints_downsampling[:, 0] - veh_pose[0]) ** 2 + (waypoints_downsampling[:, 1] - veh_pose[1]) ** 2)
        id_nearest = np.argmin(distances)
        return waypoints_downsampling[id_nearest]

    def calc_yaw_diff_two_waypoints(self, waypoint1: Tuple[float, float, float], waypoint2: Tuple[float, float, float]):
        """计算两个路径点之间的夹角,结果在[-pi,pi]范围内,"""
        angle1 = waypoint1[2]
        angle2 = waypoint2[2]
        yaw_diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
        return yaw_diff

    def get_ego_reference_path_to_goal(self, observation):
        """全局路径规划器.
        获取HD Map中ego车到达目标区域的参考路径.
        注1：该参考路径仅表示了对于道路通行规则的指示。
            数据处理来源:1)在地图中手动标记dubinspose;2)生成dubins curve;3)离散拼接.
        注2:显然该参考路径存在以下缺陷--
            1)在实际道路中使用该参考路径跟车行驶时不符合曲率约束;2)onsite_mine仿真中存在与边界发生碰撞的风险.
        注3:onsite_mine自动驾驶算法设计鼓励参与者设计符合道路场景及被控车辆特性的实时轨迹规划算法.

        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径(拼接后).
        """
        if 0:  # 仅适用于 会车场景 jiangtong-9-1-1,不在比赛场景中
            ego_connected_path_tokens = ["path-107", "path-93"]
            return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)
        #################更新参数#################
        self._ego_x = observation["vehicle_info"]["ego"]["x"]
        self._ego_y = observation["vehicle_info"]["ego"]["y"]
        self._ego_v = observation["vehicle_info"]["ego"]["v_mps"]
        self._ego_yaw = observation["vehicle_info"]["ego"]["yaw_rad"]

        #################定位主车和目标点所在几何#################
        ego_polygon_token = observation["hdmaps_info"]["tgsc_map"].get_polygon_token_using_node(self._ego_x, self._ego_y)
        ego_polygon_id = int(ego_polygon_token.split("-")[1])

        #################获取目标车最匹配的path-token #################
        ego_path_token, ego_path_id = self.get_best_matching_path_token_from_polygon(
            (self._ego_x, self._ego_y, self._ego_yaw), ego_polygon_token, observation
        )

        # 广度优先搜索,一般搜索不超过3层；
        # todo 可以使用标准的树搜索方式重写.
        path_connect_tree = {"layer_1": ego_path_token, "layer_2": {}}
        second_layer_path_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[ego_path_id]["outgoing_tokens"]

        # 第二层
        for _, token_2 in enumerate(second_layer_path_tokens):  # path_token_2
            id_2 = int(token_2.split("-")[1])  # path_id_2
            if token_2 not in path_connect_tree["layer_2"]:
                path_connect_tree["layer_2"][token_2] = {
                    "flag_inside_goal_area": False,
                    "layer_3": {},
                }
            ref_path = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_2]["waypoints"])
            # 搜索终止条件:参考路径的waypoints有点waypoint在目标区域内部
            flag_inside_goal_area = self.has_waypoint_inside_goal_area(
                ref_path,
                observation["test_setting"]["goal"]["x"],
                observation["test_setting"]["goal"]["y"],
            )
            if flag_inside_goal_area:
                path_connect_tree["layer_2"][token_2]["flag_inside_goal_area"] = True
                ego_connected_path_tokens = [path_connect_tree["layer_1"], token_2]
                return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)
            else:
                path_connect_tree["layer_2"][token_2]["flag_inside_goal_area"] = False
                outgoing_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[id_2]["outgoing_tokens"]
                for _, token in enumerate(outgoing_tokens):
                    if token not in path_connect_tree["layer_2"][token_2]["layer_3"]:
                        path_connect_tree["layer_2"][token_2]["layer_3"][token] = {}

        # 第三层
        for _, token_2 in enumerate(second_layer_path_tokens):
            thrid_layer_path_tokens = path_connect_tree["layer_2"][token_2]["layer_3"]
            for _, token_3 in enumerate(thrid_layer_path_tokens):
                id_3 = int(token_3.split("-")[1])
                if not path_connect_tree["layer_2"][token_2]["layer_3"][token_3]:  # 空
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3] = {
                        "flag_inside_goal_area": False,
                        "layer_4": {},
                    }
                ref_path = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_3]["waypoints"])
                flag_inside_goal_area = self.has_waypoint_inside_goal_area(
                    ref_path,
                    observation["test_setting"]["goal"]["x"],
                    observation["test_setting"]["goal"]["y"],
                )
                if flag_inside_goal_area:
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["flag_inside_goal_area"] = True
                    ego_connected_path_tokens = [
                        path_connect_tree["layer_1"],
                        token_2,
                        token_3,
                    ]
                    return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)

                else:
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["flag_inside_goal_area"] = False
                    outgoing_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[id_3]["outgoing_tokens"]
                    for _, token in enumerate(outgoing_tokens):
                        if token not in path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["layer_4"]:
                            # path_connect_tree['layer_2']['path-7']['layer_3']['path-70']['layer_4']
                            path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["layer_4"][token] = {}

    @staticmethod
    def has_waypoint_inside_goal_area(
        ref_path_waypoints: np.array = None,
        goal_area_x: List = None,
        goal_area_y: List = None,
    ) -> bool:
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

    def get_connected_waypoints_from_multi_path(self, observation, connected_path_tokens: List = None):
        """获得多条路径拼接后的waypoints. waypoints(x,y,yaw,heigh,slope).

        Args:
            observation (_type_): 环境信息.
            connected_path_tokens (List, optional): _description_. Defaults to None.

        Returns:
            List: 多条路径拼接后的waypoints(x,y)
        """

        connected_waypoints = []

        connected_waypoints = [
            point
            for token_path in connected_path_tokens
            for point in observation["hdmaps_info"]["tgsc_map"].reference_path[int(token_path.split("-")[1])]["waypoints"]
        ]

        return connected_waypoints

    def get_connected_waypoints_from_multi_path_array(self, observation, connected_path_tokens: List = None):
        """获得多条路径拼接后的waypoints.使用np.array

        Args:
            connected_path_tokens (List, optional): _description_. Defaults to None.
        """
        # Initialize connected_waypoints as None
        connected_waypoints = None
        for token_path in connected_path_tokens:
            id_path = int(token_path.split("-")[1])

            # Get temp_waypoints from observation
            temp_waypoints = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_path]["waypoints"])

            # Check if connected_waypoints is None, if so assign temp_waypoints to it
            # otherwise concatenate temp_waypoints to connected_waypoints
            if connected_waypoints is None:
                connected_waypoints = temp_waypoints
            else:
                connected_waypoints = np.concatenate((connected_waypoints, temp_waypoints), axis=0)

        return connected_waypoints
