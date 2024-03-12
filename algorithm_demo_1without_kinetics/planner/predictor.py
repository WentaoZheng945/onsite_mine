  # 内置库 
import sys
import os
import math
import importlib


# 第三方库
import numpy as np
import bisect
from typing import Dict,List,Tuple,Optional,Union


# 导入onsite-mine相关模块（库中含有-,因此使用动态importlib动态导入包）
## onsite-mine.dynamic_scenes模块
observation = importlib.import_module("onsite-mine.dynamic_scenes.observation")
## onsite-mine.common模块
utils = importlib.import_module("onsite-mine.common.utils")



class Predictor:
    """轨迹模块，针对背景车(周围障碍物车辆)轨迹预测器;"""

    def __init__(self, time_horizon: float = 1.5) -> None:
        """默认构造 Predictor.
        time_horizon: 预测时域,默认为5s .
        """
        self.prediction_total_time_ = time_horizon
        self.veh_traj_predicted = {}
        self.traj_future_assesment = {}
        self.veh_predictor_info = dict()

    def predict(
        self,
        observation: observation = None,
        traj: List = None,
        predictor_type="CACV_PREDICTOR",
    ) -> List[Dict]:
        """调用轨迹预测器,进行目标车辆轨迹预测.

        :traj:待预测的背景车的轨迹;
        :observation :场景实时的观察结果;
        :veh_traj_predicted : 返回值,目标车辆未来轨迹预测信息;
        """
        self.predictor_type_ = predictor_type
        t_now_ = round(float(observation["test_setting"]["t"]), 2)
        interval_ = observation["test_setting"]["dt"]
        max_t_ = observation["test_setting"]["max_t"]

        # 当前时刻所有背景车的状态信息
        self.veh_traj_predicted = {}  # 每一个step,所有的veh_traj_predicted都清空
        for id_veh in observation["vehicle_info"].keys():
            if id_veh == "ego":
                continue
            self.veh_traj_predicted.update({id_veh: {}})
            self.veh_traj_predicted[id_veh][-1] = {}
            self.veh_traj_predicted[id_veh]["shape"] = traj[id_veh]["shape"]
            self.veh_traj_predicted[id_veh][str(t_now_)] = traj[id_veh][str(t_now_)]

        # 不同的背景车可以使用不同的预测器 ######################################
        if self.predictor_type_ == "CACV_PREDICTOR":
            # 预测多个车的轨迹,使用简单的预测器
            for id_veh in observation["vehicle_info"].keys():
                if id_veh == "ego":
                    continue
                CACV_Predictor_ = ConstantAngularVelocityPredictor(t_now_, interval_, max_t_, self.prediction_total_time_)
                traj_future_ = CACV_Predictor_.predict(traj[id_veh], t_now_)  # 调用单个车辆预测器
                for key, value in traj_future_.items():  # 遍历所有键值对
                    self.veh_traj_predicted[id_veh][key] = value
                self.traj_future_assesment[id_veh] = "ONE_SECOND"

        elif self.predictor_type_ == "LANE_SEQUENCE_PREDICTOR":
            # 预测多个车的轨迹,融合预测器
            for id_veh in observation["vehicle_info"].keys():
                if id_veh == "ego":
                    continue

                veh_x = observation["vehicle_info"][id_veh]["x"]
                veh_y = observation["vehicle_info"][id_veh]["y"]
                veh_yaw = observation["vehicle_info"][id_veh]["yaw_rad"]
                veh_pose = (veh_x, veh_y, veh_yaw)

                # 获取目标车属于哪个polygon
                polygon_token = observation["hdmaps_info"]["tgsc_map"].get_polygon_token_using_node(veh_x, veh_y)
                if observation["test_setting"]["scenario_name"] == "jiangtong_intersection_1_7_5":  # !先特殊处理
                    if id_veh == 1 and polygon_token == "polygon-0":
                        polygon_token = "polygon-8"
                id_polygon = int(polygon_token.split("-")[1])

                # 获取目标车最匹配的dubinspose(在多边形所属的link_dubinspose_tokens中查找)
                polygon_info = observation["hdmaps_info"]["tgsc_map"].polygon[id_polygon]
                if polygon_info["type"] == "road":
                    token_dubinspose = observation["hdmaps_info"]["tgsc_map"].get_dubinspose_token_from_polygon(veh_pose, polygon_token)
                    id_dubinspose = int(token_dubinspose.split("-")[1])
                else:
                    # !先特殊处理
                    # token_dubinspose = observation['hdmaps_info']['tgsc_map'].get_dubinspose_token_from_polygon(veh_pose,polygon_token)
                    # id_dubinspose = int( token_dubinspose.split('-')[1])
                    token_dubinspose = "None_inner_intersection"
                # 确定要匹配的 dubinspose start
                if id_veh not in self.veh_predictor_info:  # 第一次 出现id_veh车 & 第一次匹配到dubinspose,以后一直使用该dubinspose start
                    token_dubinspose_start = token_dubinspose
                    id_dubinspose_start = id_dubinspose
                    # 特殊处理1
                    if observation["test_setting"]["scenario_name"] == "jiangtong_intersection_1_7_5" and id_veh == 1:
                        token_dubinspose_start = "dubinspose-10"
                        id_dubinspose_start = 10
                    # 特殊处理2
                    if observation["test_setting"]["scenario_name"] == "jiangtong_intersection_1_7_5" and id_veh == 3:
                        token_dubinspose_start = "dubinspose-6"
                        id_dubinspose_start = 6
                    # 匹配多边形,获取参考路径;
                    ref_path_tokens = observation["hdmaps_info"]["tgsc_map"].dubins_pose[id_dubinspose_start]["link_connectivity_path_tokens"]
                    veh_match_info_first = {
                        "match_info_time": observation["test_setting"]["t"],
                        "polygon_token": polygon_token,
                        "token_dubinspose": token_dubinspose_start,
                        "ref_path_tokens": ref_path_tokens,
                    }  # ! 注意:ref_path_tokens 确定后不可更改
                    self.veh_predictor_info.update({id_veh: {}})  # 新创建一个
                    self.veh_predictor_info[id_veh]["veh_match_info_first"] = veh_match_info_first
                else:
                    token_dubinspose_start = self.veh_predictor_info[id_veh]["veh_match_info_first"]["token_dubinspose"]
                    id_dubinspose_start = int(token_dubinspose_start.split("-")[1])
                # 计算车辆定位点与某个dubinspose纵向位姿关系:前后
                _, longitudinal_error, _ = utils.compute_two_pose_error(
                    veh_x,
                    veh_y,
                    veh_yaw,
                    observation["hdmaps_info"]["tgsc_map"].dubins_pose[id_dubinspose_start]["x"],
                    observation["hdmaps_info"]["tgsc_map"].dubins_pose[id_dubinspose_start]["y"],
                    observation["hdmaps_info"]["tgsc_map"].dubins_pose[id_dubinspose_start]["yaw"],
                )
                # 切换预测器类型----针对车辆id_veh
                if longitudinal_error >= -0.8:  # 条件1:进入交叉路口,很接近 dubinspose-start;条件2:离开交叉路口,接近 dubinspose-end
                    predictor_type_temp = "LANE_SEQUENCE_PREDICTOR"
                else:
                    predictor_type_temp = "CACV_PREDICTOR"

                if "predictor_type_update" not in self.veh_predictor_info[id_veh]:  # 第一次 出现id_veh车的"predictor_type_update"
                    self.veh_predictor_info[id_veh].update({"predictor_type_update": predictor_type_temp})
                self.veh_predictor_info[id_veh]["predictor_type_update"] = predictor_type_temp

                if self.veh_predictor_info[id_veh]["predictor_type_update"] == "CACV_PREDICTOR":
                    # 5.0s全部使用 CACV_PREDICTOR预测器
                    if "CACV_PREDICTOR" not in self.veh_predictor_info[id_veh]:  # 第一次 给id_veh车初始化CACV_PREDICTOR预测器
                        self.veh_predictor_info[id_veh]["CACV_PREDICTOR"] = ConstantAngularVelocityPredictor(
                            t_now_, interval_, max_t_, self.prediction_total_time_
                        )
                    traj_future_ = self.veh_predictor_info[id_veh]["CACV_PREDICTOR"].predict(traj[id_veh], t_now_)
                    for key, value in traj_future_.items():  # 遍历所有键值对
                        self.veh_traj_predicted[id_veh][key] = value
                    self.traj_future_assesment[id_veh] = "OVER_ONE_SECOND"

                elif self.veh_predictor_info[id_veh]["predictor_type_update"] == "LANE_SEQUENCE_PREDICTOR":
                    if "LANE_SEQUENCE_PREDICTOR" not in self.veh_predictor_info[id_veh]:  # 第一次 给id_veh车初始化LANE_SEQUENCE_AND_CACV预测器
                        self.veh_predictor_info[id_veh]["LANE_SEQUENCE_PREDICTOR"] = LaneSequencePredictor(
                            t_now_, interval_, max_t_, self.prediction_total_time_
                        )

                    # 车辆匹配信息实时更新(部分1)
                    veh_match_info_update = {
                        "match_info_time": observation["test_setting"]["t"],
                        "polygon_token": polygon_token,
                        "token_dubinspose": token_dubinspose,
                    }
                    # "ref_path_tokens":self.veh_predictor_info[id_veh]['veh_match_info_first']['ref_path_tokens']
                    # ! 注意:ref_path_tokens 确定后不可更改
                    # 先获取该车未来1.5秒的预测轨迹,较准确
                    CACV_Predictor_ = ConstantAngularVelocityPredictor(t_now_, interval_, max_t_, time_horizon=1.5)
                    traj_future_1s = CACV_Predictor_.predict(traj[id_veh], t_now_)  # 修改为1.5s
                    # 调用 LANE_SEQUENCE_PREDICTOR
                    (
                        traj_future_,
                        traj_assesment_,
                        paths_prob_info,
                        flag_preview_dis_over_end,
                    ) = self.veh_predictor_info[
                        id_veh
                    ]["LANE_SEQUENCE_PREDICTOR"].predict(
                        self.veh_traj_predicted[id_veh][str(t_now_)],
                        traj_future_1s,
                        observation,
                        self.veh_predictor_info[id_veh]["veh_match_info_first"],
                        veh_match_info_update,
                    )
                    # 记录预测器结果: 车辆匹配信息实时更新(部分2----paths_prob_info)
                    veh_match_info_update["paths_prob_info"] = paths_prob_info
                    veh_match_info_update["flag_preview_dis_over_end"] = flag_preview_dis_over_end
                    self.veh_predictor_info[id_veh]["veh_match_info_update"] = veh_match_info_update
                    if flag_preview_dis_over_end == True:  # 参考路径信息太短,使用 CACV_PREDICTOR预测器
                        if "CACV_PREDICTOR" not in self.veh_predictor_info[id_veh]:  # 第一次 给id_veh车初始化CACV_PREDICTOR预测器
                            self.veh_predictor_info[id_veh]["CACV_PREDICTOR"] = ConstantAngularVelocityPredictor(
                                t_now_,
                                interval_,
                                max_t_,
                                self.prediction_total_time_,
                            )
                        traj_future_ = {}
                        traj_future_ = self.veh_predictor_info[id_veh]["CACV_PREDICTOR"].predict(traj[id_veh], t_now_)
                        for key, value in traj_future_.items():  # 遍历所有键值对
                            self.veh_traj_predicted[id_veh][key] = value
                        self.traj_future_assesment[id_veh] = "OVER_ONE_SECOND"
                    else:
                        # 记录预测器结果:预测轨迹信息实时更新
                        for key, value in traj_future_.items():  # 遍历所有键值对
                            self.veh_traj_predicted[id_veh][key] = value
                        # 记录预测器结果:预测轨迹评价
                        if self.prediction_total_time_ > 1:
                            if traj_assesment_ == "LANE_SEQ":
                                self.traj_future_assesment[id_veh] = "OVER_ONE_SECOND"
                            else:  # CACV_NOT_LANE_SEQ
                                self.traj_future_assesment[id_veh] = "ONE_SECOND"
                        elif self.prediction_total_time_ == 1:
                            self.traj_future_assesment[id_veh] = "ONE_SECOND"
                        else:
                            assert "Error:prediction 时域不支持"

                else:
                    raise Exception("###Exception### predictor_type_update异常,请检查!")

        else:
            print("### log ### error:predictor_type有问题!")

        return self.veh_traj_predicted, self.traj_future_assesment


class ConstantAngularVelocityPredictor:
    """利用纵向速度和角速度从当前位置推算出一条曲线
    extrapolates a curved line from current position using linear and angular velocity
    """

    def __init__(
        self,
        t_now: float = 0.0,
        interval: float = 0.1,
        max_t: float = 18.0,
        time_horizon: float = 1.5,
    ) -> None:
        """默认构造 获取每个场景后,进行初始化
        默认时间间隔0.1, 还有一些时间间隔是0.04;
        """
        self.prediction_total_time_ = time_horizon
        self.dt_ = interval
        self.t_now_ = t_now
        self.max_t_ = max_t
        self.traj_predict_ = dict()

    def predict(self, traj_all: Dict, t_now_: float) -> Dict:
        t_now_str = str(t_now_)
        self.t_now_ = t_now_  #! 切忌要更新该值
        t = t_now_
        dt = self.dt_
        x = traj_all[t_now_str]["x"]
        y = traj_all[t_now_str]["y"]
        v = traj_all[t_now_str]["v_mps"]
        yaw = traj_all[t_now_str]["yaw_rad"]
        acc = traj_all[t_now_str]["acc_mpss"]
        yaw_rate = traj_all[t_now_str]["yawrate_radps"]

        delta_t = self.max_t_ - self.t_now_
        if delta_t < self.prediction_total_time_:
            self.numOfTrajPoint_ = int(delta_t / self.dt_)
        else:
            self.numOfTrajPoint_ = int(self.prediction_total_time_ / self.dt_)  # 5秒*10hz;

        for i in range(self.numOfTrajPoint_):
            t += dt
            x += dt * (v * np.cos(yaw) + acc * np.cos(yaw) * 0.5 * dt + yaw_rate * v * np.sin(yaw) * 0.5 * dt)  # 确定yaw定义
            y += dt * (v * np.sin(yaw) + acc * np.sin(yaw) * 0.5 * dt + yaw_rate * v * np.cos(yaw) * 0.5 * dt)
            yaw += dt * yaw_rate
            v += dt * acc

            if self.dt_ < 0.1:
                str_time = str(round(t, 3))
            else:
                str_time = str(round(t, 2))
            self.traj_predict_[str_time] = {
                "x": round(x, 2),
                "y": round(y, 2),
                "v": round(v, 2),
                "a": round(acc, 2),
                "yaw": round(yaw, 3),
            }

        return self.traj_predict_


class LaneSequencePredictor:
    """
    沿着车道序列的拓扑关系进行5s轨迹生成;预测轨迹靠近车道中心线,也做了平滑.
    1、获取目标车辆进入路口的 未来可行驶车道(路径)信息;
    2、构建评价函数,做概率预测(只给出最大可能性的path 预测)
    3、确定path后,纵向速度预测按照匀速假设
    4.结果经过EKF进行过滤平滑;
    """

    def __init__(self, t_now=0.0, interval=0.1, max_t=18.0, time_horizon=5) -> None:
        """默认构造 获取每个场景后,进行初始化
        默认时间间隔0.1;
        """
        self.prediction_total_time_ = time_horizon
        self.dt_ = interval
        self.t_now_ = t_now
        self.max_t_ = max_t
        self.road_id_max_prob_last_ = -1
        self.veh_traj_predicted = dict()
        self.veh_match_info_first = dict()
        self.veh_match_info_update = dict()

    def predict(
        self,
        traj_now: Dict,
        traj_future_1s: Dict,
        observation: observation,
        veh_match_info_first: Dict,
        veh_match_info_update: Dict,
    ):
        """调用轨迹预测器,进行目标车辆轨迹预测
        veh_match_info_update 是相比较 veh_match_info_first 实时更新的部分
        """
        flag_preview_dis_over_end = False
        self.veh_match_info_first = veh_match_info_first
        self.veh_match_info_update = veh_match_info_update

        self.t_now_ = observation["test_setting"]["t"]
        dt = self.dt_
        # 1) 获取目标车辆实时状态信息:当前位姿...
        self.flag_use_CACV_for_1s = False
        self.x = traj_now["x"]
        self.y = traj_now["y"]
        self.v = traj_now["v_mps"]
        self.yaw = traj_now["yaw_rad"]
        self.acc = traj_now["acc_mpss"]
        self.yawrate = traj_now["yawrate_radps"]

        # 2) 获取目标车辆未来可行驶车道信息;
        ref_path_tokens = veh_match_info_first["ref_path_tokens"]

        # 3) 构建评价函数, 做概率预测,实时输出多个待选Path的概率排序
        paths_prob_info = self._reference_path_evaluator(traj_future_1s, observation)

        # 4) 进行概率意图预测的使用——用于长时轨迹预测,可以提出不同的方案
        # ! 方案1:依据当前path点,进行纵向恒速推断;获取5s处的预测轨迹点;直接进行三次多项式曲线插值
        # 4.1) 选取最大概率参考路径,获取5秒推断目标点;匀速推断
        ref_path_token = paths_prob_info[0][0]
        ref_path_id = int(ref_path_token.split("-")[1])
        preview_dis = self.v * self.prediction_total_time_
        path_points = observation["hdmaps_info"]["tgsc_map"].reference_path[ref_path_id]["pathsample"]["path_points"]
        target_path_point_id, flag_preview_dis_over_end = utils.find_preview_point_index_over_end(self.x, self.y, preview_dis, path_points)
        if flag_preview_dis_over_end == True:  # 目标车辆太靠近 dubinspose end,需要返回flag,调用CACV预测器
            traj_assesment_ = "LANE_SEQ"
            return (
                self.veh_traj_predicted,
                traj_assesment_,
                paths_prob_info,
                flag_preview_dis_over_end,
            )

        # 4.2)进行多项式轨迹生成
        pt1 = [self.x, self.y]
        pt2 = [
            path_points[target_path_point_id][0],
            path_points[target_path_point_id][1],
        ]
        v1 = [self.v * np.cos(self.yaw), self.v * np.sin(self.yaw)]
        v2 = [
            self.v * np.cos(path_points[target_path_point_id][2]),
            self.v * np.sin(path_points[target_path_point_id][2]),
        ]
        a1 = [0.0, 0.0]
        a2 = [0.0, 0.0]
        interpolator = utils.PolynomialInterpolation(pt1, pt2, v1, v2, a1, a2)
        interpolation_trajectory = interpolator.quintic_polynomial_interpolation()
        # 提取插值结果的 x 和 y 坐标
        x_values = [point[0] for point in interpolation_trajectory]
        y_values = [point[1] for point in interpolation_trajectory]

        # TODO 编写生成轨迹的函数
        t = self.t_now_
        for i in range(len(interpolation_trajectory)):
            str_time = str(round(t, 2))
            t += dt
            x = interpolation_trajectory[i][0]
            y = interpolation_trajectory[i][1]
            v = self.v
            acc = self.acc
            yaw = self.yaw
            self.veh_traj_predicted[str_time] = {
                "x": round(x, 2),
                "y": round(y, 2),
                "v": round(v, 2),
                "a": round(acc, 2),
                "yaw": round(yaw, 3),
            }

        # TODO 方案2:参考路径曲率信息+ EKF

        # TODO 方案3:如果实时位置与最近点距离太远,xxxx

        traj_assesment_ = "LANE_SEQ"
        return (
            self.veh_traj_predicted,
            traj_assesment_,
            paths_prob_info,
            flag_preview_dis_over_end,
        )

    def _reference_path_evaluator(self, traj_future_1s: Dict, observation: observation) -> list:
        """构建评价函数,做概率预测,输出多个待选Path的概率排序

        Args:
            traj_future_1s (Dict):短时1.5s轨迹预测结果;

        Returns:
            Dict:多个待选Path的概率排序
        """
        paths_prob_info_temp = list()
        similarity_score = list()
        sum_value = 0

        # 计算相似度值
        for path_token in self.veh_match_info_first["ref_path_tokens"]:
            nearest_id, score, similarity_euclidean_distance, similarity_deltaYaw = self._cal_similarity_score(
                path_token, traj_future_1s, observation
            )
            similarity_score.append(
                [
                    path_token,
                    nearest_id,
                    score,
                    similarity_euclidean_distance,
                    similarity_deltaYaw,
                    0.0,
                ]
            )
            sum_value += score

        # 计算概率
        for i in range(len(similarity_score)):
            similarity_score[i][5] = similarity_score[i][2] / sum_value
        # 使用 lambda 表达式作为比较函数,按照数字由大到小进行排序
        paths_prob_info_temp = sorted(similarity_score, key=lambda x: x[2], reverse=True)

        return paths_prob_info_temp

    def _cal_similarity_score(self, path_token: str, traj_future_1s: dict, observation: observation):

        # TODO:先验条件1——筛选掉block掉的路径
        # TODO:先验知识2——根据车大小及路宽度筛选掉大车小路;
        w1 = 0.02  #! 可调参数--欧式距离权重,距离越小越相似
        w2 = 0.98  #! 可调参数--航向角差权重,角度差越小越相似
        path_id = int(path_token.split("-")[1])
        path_points = observation["hdmaps_info"]["tgsc_map"].reference_path[path_id]["pathsample"]["path_points"]
        nearest_id = utils.find_nearest_point_index(self.x, self.y, path_points)

        # 根据1s的轨迹预测点,预瞄N个最近路径点,计算航向角差均值
        similarity_score_deltaYaw_list = []
        for _, value in traj_future_1s.items():
            nearest_id = utils.find_nearest_point_index(value["x"], value["y"], path_points)
            similarity_score_deltaYaw_list.append(self._cal_similarity_score_two_yaw(value["yaw"], path_points[nearest_id][2]))

        similarity_deltaYaw = np.mean(similarity_score_deltaYaw_list)

        # 距离相似性,当前只计算了实时位置点;后续改成短时预测点的均值
        euclidean_distance = math.sqrt((self.x - path_points[nearest_id][0]) ** 2 + (self.y - path_points[nearest_id][1]) ** 2)
        lane_width = 3.5  # 取值半个车道宽1.75米,市区道路一般 2.75米到3.25米;矿区按照3.5米
        if euclidean_distance > lane_width:
            euclidean_distance = lane_width
        similarity_euclidean_distance = abs(self._linear_map_to_range(euclidean_distance, start1=0, stop1=lane_width, start2=1, stop2=0))

        similarity_score = w1 * similarity_euclidean_distance + w2 * similarity_deltaYaw  # 距离相似性,偏航相似性

        return (
            nearest_id,
            similarity_score,
            similarity_euclidean_distance,
            similarity_deltaYaw,
        )

    @staticmethod
    def _cal_similarity_score_two_yaw(yaw1_rad: float, yaw2_rad: float) -> float:
        """计算两个角度之间的相似性得分,结果在[0,1] 范围内;
        yaw1_rad,yaw2_rad 取值[0,2pi),无顺序要求
        """

        delt_yaw_rad = yaw1_rad - yaw2_rad
        delt_yaw_rad = (delt_yaw_rad + 2 * np.pi) % (2 * np.pi)  # 将偏差限制在 [0,360) 范围内
        delt_yaw_similarity = abs(np.cos(0.5 * delt_yaw_rad))  # 第2种映射方法:将角度差[0,2pi) 映射到 [0,1] 范围内

        return delt_yaw_similarity

    @staticmethod
    def _linear_map_to_range(value: float, start1: float, stop1: float, start2: float, stop2: float) -> float:
        """value,由start1,stop1 线性映射到start2,stop2

        Args:
            value (float):转化前的值
            start1 (float):映射前范围[start1,stop1]
            stop1 (float):  映射前范围[start1,stop1]
            start2 (float): 映射后范围[start2,stop2]
            stop2 (float):映射后范围[start2,stop2]

        Returns:
            float:转化后的值
        """

        scaled_value = (value - start1) / (stop1 - start1)
        mapped_value = start2 + scaled_value * (stop2 - start2)
        return mapped_value

    def _generate_path_point_S(self, ref_path):
        """路径点之间增加累加距离计算

        输入:
        ----------
        ref_path:当前场景下最大概率的一条path,已经拼接好,包含 [x,y,heading_rad_KF,heading_rad]

        输出:
        ----------
        ref_path_XYHeadingS :增加累加距离 [x,y,heading_rad_KF,heading_rad,s]
        """
        # umOfPathPoint = len(ref_path)
        # s_distance_list =[]
        ref_path_XYHeadingS = []
        sum_dist = 0

        for i, subarray in enumerate(ref_path):
            if i > 0:
                delta_dist = np.sqrt((ref_path[i][0] - ref_path[i - 1][0]) ** 2 + (ref_path[i][1] - ref_path[i - 1][1]) ** 2)
            else:
                delta_dist = 0
            sum_dist += delta_dist
            ref_path_XYHeadingS.append(np.concatenate((subarray, np.array([sum_dist])), axis=0))

        return ref_path_XYHeadingS

    def _generate_traj_predicted(self, ref_path_XYHeadingS):
        """纵向匀速假设"""
        t = self.t_now_
        dt = self.dt_
        v = self.v

        delta_t = self.max_t_ - self.t_now_
        if delta_t < self.prediction_total_time_:
            self.numOfTrajPoint_ = int(delta_t / self.dt_)
        else:
            self.numOfTrajPoint_ = int(self.prediction_total_time_ / self.dt_)  # 5秒*10hz;

        dist_predic_space_max = (
            ref_path_XYHeadingS[-1][4] - ref_path_XYHeadingS[self.road_nearest_point_index][4]
        )  # 当前时刻该 ref_path 还剩下的距离,作为最大预测距离
        self.traj_predict_ = {}
        for i in range(self.numOfTrajPoint_):
            t += dt
            delta_t = t - self.t_now_  # 累加时间
            sum_distance_1 = delta_t * v  # 预计行驶距离
            if sum_distance_1 > dist_predic_space_max:
                sum_distance_1 = dist_predic_space_max

            if self.dt_ < 0.1:
                str_time = str(round(t, 3))
            else:
                str_time = str(round(t, 2))

            self.traj_predict_[str_time] = self._generate_traj_predicted_sub(
                ref_path_XYHeadingS,
                sum_distance_1 + ref_path_XYHeadingS[self.road_nearest_point_index][4],
            )

        return self.traj_predict_

    def _generate_traj_predicted_sub(self, ref_path_XYHeadingS, sum_distance):
        """依据三个参数,求解单个预测点位姿(状态信息)
        self.road_nearest_point_index,
        ref_path_XYHeadingS : [x,y,heading_rad_KF,heading_rad,s]
        sum_distance :距离
        """
        s_distance_list = [arr[4] for arr in ref_path_XYHeadingS]
        prev_index_, next_index_, ratio_ = self._get_index_and_ratio(s_distance_list, sum_distance)
        x = ref_path_XYHeadingS[prev_index_][0] + ratio_ * (ref_path_XYHeadingS[next_index_][0] - ref_path_XYHeadingS[prev_index_][0])
        y = ref_path_XYHeadingS[prev_index_][1] + ratio_ * (ref_path_XYHeadingS[next_index_][1] - ref_path_XYHeadingS[prev_index_][1])
        yaw = ref_path_XYHeadingS[prev_index_][2] + ratio_ * (ref_path_XYHeadingS[next_index_][2] - ref_path_XYHeadingS[prev_index_][2])
        v = self.v
        acc = 0
        yaw = 0

        traj_pred_sub_ = {
            "x": round(x, 2),
            "y": round(y, 2),
            "v": round(v, 2),
            "a": round(acc, 2),
            "yaw": round(yaw, 3),
        }

        return traj_pred_sub_

    def _get_index_and_ratio(self, sorted_list, distance):
        """由小到大排列的距离列表,输入距离值,获得索引"""
        # 寻找距离值在列表中的插入位置
        index = bisect.bisect_left(sorted_list, distance)

        # 处理特殊情况:距离值小于列表中的最小值
        if index == 0:
            return 0, 0, 0

        # 处理特殊情况:距离值大于等于列表中的最大值
        if index == len(sorted_list):
            return len(sorted_list) - 1, len(sorted_list) - 1, 1

        # 获取距离值的前后索引值和线性差值比例,当前值到前一个值的距离的占比
        prev_index = index - 1
        next_index = index
        prev_value = sorted_list[prev_index]
        next_value = sorted_list[next_index]
        ratio = (distance - prev_value) / (next_value - prev_value)

        return prev_index, next_index, ratio
   
    


if __name__ == "__main__":
    import time
    # 调试代码