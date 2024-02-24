# 内置库 
import math
import os
import sys
import importlib

# 第三方库
import numpy as np

# 导入onsite-mine相关模块（库中含有-,因此使用动态importlib动态导入包）
## onsite-mine.common模块
kdtree = importlib.import_module("onsite-mine.common.kdtree")

 
class Point_t:
    """2维k-d tree的基础数据结构,KDNode."""
    def __init__(self, x:float=None, y:float=None, ind:int=None):
        """_summary_

        Args:
            x (float, optional): 笛卡尔坐标系x. Defaults to None.
            y (float, optional): 笛卡尔坐标系y. Defaults to None.
            ind (int, optional): 在data中的结果索引inddex. Defaults to None.
        """
        self.x = x
        self.y = y
        self.ind = ind




class Item(object):
    def __init__(self,x,y,data):
        self.coords = (x,y)
        self.data = data
    def __len__(self):
        return len(self.coords)
    def __getitem__(self,i):
        return self.coords[i]
    def __repr__(self):
        return 'Item({},{},{})'.format(self.coords[0],self.coords[1],self.data)




class MotionController:
    """运动控制模块.
        功能设计:跟踪规划模块输入的路径曲线,速度曲线;
    """
    def __init__(self):
        self._min_prev_dist = 14.0 #最小预瞄距离
        self._prev_k = 1.5 #预瞄距离速度系数
        self.L = 5.6 #轴距
        self._is_create_kdtree = False
        self._projection_vehicle_S = list() #存储投影车
        self._last_acc = -999
        

    def process(self,vehicle_info,path_waypoints,traj_future,observation):
        """控制主函数
        输入:_a_max自车最大加速度,_vel自车当前车速,_vel0自车期望车速,delta加速度指数,
        _dv自车与前车速度差,_s自车与前车车距,_b舒适减速度,_t安全时距,_s0最小车距
        """
        #################更新参数#################
        self._ego_v = vehicle_info['ego']['v_mps']
        self._ego_x = vehicle_info['ego']['x']
        self._ego_y = vehicle_info['ego']['y']
        self._ego_yaw = vehicle_info['ego']['yaw_rad']
        self._ego_shape = vehicle_info['ego']['shape']
        self._vehicle_info = vehicle_info
        self._projection_vehicle_S.clear()

        #################只创建一次kdtree#################
        if self._is_create_kdtree == False:
            path_tuple = list()
            i=0 #记录路径点下标
            for waypoint in path_waypoints:
                point = Item(waypoint[0],waypoint[1],str(i))#! (x,y,index) 用于构建KD-tree
                path_tuple.append(point)
                i+=1
            self._tree = kdtree.create(path_tuple)

        #################匹配与车辆最近的路径点#################
        nearest_tree_node = self._tree.search_nn([self._ego_x,self._ego_y])
        self._nearest_pt_id = int(nearest_tree_node[0].data.data) #最近路径点的下标
        self._nearest_pt = path_waypoints[self._nearest_pt_id]

        #################计算预瞄距离#################
        prev_dist = self._min_prev_dist + self._prev_k * self._ego_v

        #################寻找预瞄点#################
        prev_pt = Point_t(0,0,-10000)
        if len(path_waypoints) == 0:
            print("###log### PATH EMPTY!!!")
        for i in range(-1,-len(path_waypoints)-1,-1):
            temp_dist = math.hypot(path_waypoints[i][0]-self._ego_x,path_waypoints[i][1]-self._ego_y)
            if temp_dist < prev_dist:
                prev_pt.x = path_waypoints[i][0]
                prev_pt.y = path_waypoints[i][1]
                prev_pt.ind = 0
                break
        if prev_pt.ind  == -10000:
            print("CAN NOT FIND PREVIEW POINT!!!")

        #################纯跟踪输出前轮转角#################
        delta = self._purePursuit(prev_pt)

        #################判断前方是否有车#################
        front_vehicle = self._findFrontVehicle(vehicle_info,path_waypoints,traj_future,observation)


        #################IDM输出加速度#################
        if front_vehicle!=None:
            dv = self._ego_v-front_vehicle['v_mps']
            s = math.hypot(self._ego_x-front_vehicle['x'],self._ego_y-front_vehicle['y']) \
                -math.hypot(self._ego_shape['width'],self._ego_shape['length'])/2 \
                -math.hypot(front_vehicle['shape']['width'],front_vehicle['shape']['length'])/2
            acc = self._IDM(1,self._ego_v,8,100.0,dv,s,1.0,3.5,30)
        else:
            acc = self._IDM(1,self._ego_v,8.,100.0,0.,9999.,1.0,3.5,30)
        if self._last_acc!=-999:
            if acc - self._last_acc>1.0:acc = self._last_acc+1
            elif acc-self._last_acc<-1.0:acc = self._last_acc-1
        self._last_acc = acc
        return acc,delta

    
    def _IDM(self,a_max,vel,vel0,delta,dv,s,b,t,s0):
        """智能驾驶模型,计算加速度.[纵向跟踪控制模块]
        输入:_a_max自车最大加速度,_vel自车当前车速,_vel0自车期望车速,delta加速度指数,
        _dv自车与前车速度差,_s自车与前车车距,_b舒适减速度,_t安全时距,_s0最小车距
        """
        expected_s = vel*t + vel*dv/(2*math.sqrt(a_max*b))
        expected_s = s0+max(0.,expected_s)
        temp_a = pow(vel/vel0,delta)
        temp_b = pow(expected_s/s,2)

        return a_max*(1-temp_a-temp_b)

    
    def _purePursuit(self,prev_pt):
        """纯跟踪算法.[横向跟踪控制模块]
        输入:prev_pt——预瞄点
        输出:delta——前轮转角
        """
        lat_d = (prev_pt.y-self._ego_y)*math.cos(self._ego_yaw)\
                -(prev_pt.x-self._ego_x)*math.sin(self._ego_yaw)#横向偏差
        Ld = math.hypot(prev_pt.y-self._ego_y,prev_pt.x-self._ego_x) #车与预瞄点距离
        delta = math.atan(2*self.L*lat_d/(Ld**2)) #前轮转角
        return delta

    
    def _findFrontVehicle(self,vehicle_info,path_waypoints,traj_future,observation):
        """ego车查找前车.
        输入:vehicle_info——背景车信息,path——自车轨迹
        输出:front_vehicle——前车信息
        """
    
        ######################制作投影##########################
        for key,vehicle in vehicle_info.items():
            if key == 'ego':continue #如果是自车,跳过
            dist = math.hypot(vehicle['x']-self._ego_x,vehicle['y']-self._ego_y) #背景车与本车距离
            if dist > 50:continue #如果距离过大,跳过

            #如果背景车与路径点距离足够小,投影到路径上
            # nearest_dist = self._tree.search_nn_dist([vehicle['x'],vehicle['y']])
            nearest_dist = self._tree.search_nn([vehicle['x'],vehicle['y']])
            dist_temp =  math.hypot(vehicle['x']-nearest_dist[0].data.coords[0],vehicle['y']-nearest_dist[0].data.coords[1]) #   
            # if nearest_dist<self._ego_shape['width']:
            if dist_temp<self._ego_shape['width']:
                self._projectVehicle(vehicle,path_waypoints)
                continue

            #预测0.5、1.0、1.5s后自车位置
            predict_ego = dict()
            for i in np.arange(0.5,4,0.5):
                predict_ego_id = self._predictEgoVehicle(i,path_waypoints)#预测自车位置
                predict_ego_pt = path_waypoints[predict_ego_id]
                ego = dict()
                ego['x'] = predict_ego_pt[0]
                ego['y'] = predict_ego_pt[1]
                if predict_ego_id == len(path_waypoints)-1:
                    yaw_vector = [path_waypoints[predict_ego_id][0]-path_waypoints[predict_ego_id-1][0],
                                        path_waypoints[predict_ego_id][1]-path_waypoints[predict_ego_id-1][1]]
                else:
                    yaw_vector = [path_waypoints[predict_ego_id+1][0]-path_waypoints[predict_ego_id][0],
                                        path_waypoints[predict_ego_id+1][1]-path_waypoints[predict_ego_id][1]]
                ego['yaw_rad'] = self._calYaw(yaw_vector[0],yaw_vector[1])
                ego['width'] = self._ego_shape['width']
                ego['length'] = self._ego_shape['length']
                predict_ego[i] = ego
            
            #预测0.5、1.0、1.5s后背景车位置
            for i in np.arange(0.5,4.,0.5):
                if str(round(float(observation['test_setting']['t'])+i,2)) in traj_future[key]:
                    time_number = round(float(observation['test_setting']['t'])+i,2) # ! xxx
                    predict_vehicle = traj_future[key][str(time_number)]
                    # is_collision = self.__collisionDetect([predict_ego[i]['x'],predict_ego[i]['y'],predict_ego[i]['yaw_rad'],predict_ego[i]['width'],predict_ego[i]['length']],
                    is_collision = self.__collisionDetect([predict_ego[i]['x'],predict_ego[i]['y'],predict_ego[i]['yaw_rad'],predict_ego[i]['width'],predict_ego[i]['length']],
                        [predict_vehicle['x'],predict_vehicle['y'],predict_vehicle['yaw'],vehicle['shape']['width'],vehicle['shape']['length']])
                    if is_collision==True:#如果相撞,投影到路径线上
                        self._projectVehicle(vehicle,path_waypoints)
                        time_str_list = []
                        vehicle_to_project = dict()
                        for j in np.arange(0.5,4.,0.5):
                            # time_str_list.append(round(float(observation['test_setting']['t'])+j,2))
                            time_str = round(float(observation['test_setting']['t'])+j,2)
                            if time_str in traj_future[key]:
                                vehicle_to_project['x'] = traj_future[key][time_str]['x']
                                vehicle_to_project['y'] = traj_future[key][time_str]['y']
                                vehicle_to_project['yaw_rad'] = traj_future[key][time_str]['yaw']
                                vehicle_to_project['v_mps'] = traj_future[key][time_str]['v']
                                vehicle_to_project['a_mpss'] = traj_future[key][time_str]['a']
                                self._projectVehicle(vehicle_to_project,path_waypoints)
                            else:break
                        break
        ######################寻找最近前车##########################
        front_vehicle = None
        front_vehicle_match_id = 999999999
        for vehicle in  self._projection_vehicle_S:
            if vehicle['match_path_point_id'] <= self._nearest_pt_id:#在本车后方,跳过
                continue
            else:
                if vehicle['match_path_point_id']<front_vehicle_match_id:
                    s = math.hypot(self._ego_x-vehicle['x'],self._ego_y-vehicle['y']) \
                        -math.hypot(self._ego_shape['width'],self._ego_shape['length'])/2 \
                        -math.hypot(vehicle['shape']['width'],vehicle['shape']['length'])/2
                    if s > 0:
                        front_vehicle = vehicle
        return front_vehicle

                
    def _predictEgoVehicle(self,t,path_waypoints):
        """预测T时间后自车位置(在待行驶路径上进行推算).
        输入:t——预测时间,path——自车轨迹
        输出:预测位置路径点下标
        """
        s = self._ego_v*t #行驶路程
        drive_s = 0.0
        waypoint = self._nearest_pt
        predict_pt_id = self._nearest_pt_id
        while drive_s<s:
            drive_s += math.hypot(waypoint[0]-path_waypoints[predict_pt_id+1][0],waypoint[1]-path_waypoints[predict_pt_id+1][1])
            predict_pt_id+=1
            waypoint = path_waypoints[predict_pt_id]
        return predict_pt_id
            
    
    def _projectVehicle(self,vehicle,path_waypoints):
        """将车辆BOX投影到路径上.
        输入:vehicle——车辆,path——路径
        """
        nearest_p_id = int(self._tree.search_nn([vehicle['x'],vehicle['y']])[0].data.data)

        project_vehicle = dict() #投影车辆
        project_vehicle['x'] = path_waypoints[nearest_p_id][0]
        project_vehicle['y'] = path_waypoints[nearest_p_id][1]
        project_vehicle['shape']=dict()
        project_vehicle['shape']['width'] = vehicle['shape']['width']
        project_vehicle['shape']['length'] = vehicle['shape']['length']

        #project_vehicle['yawrate_rads'] = vehicle['yawrate_rads']
        project_vehicle['match_path_point_id'] = nearest_p_id

        #################计算投影轴#################
        if nearest_p_id == len(path_waypoints)-1:
            projection_axis = [path_waypoints[nearest_p_id][0]-path_waypoints[nearest_p_id-1][0],
                                path_waypoints[nearest_p_id][1]-path_waypoints[nearest_p_id-1][1]]
        else:
            projection_axis = [path_waypoints[nearest_p_id+1][0]-path_waypoints[nearest_p_id][0],
                                path_waypoints[nearest_p_id+1][1]-path_waypoints[nearest_p_id][1]]
        #################计算车辆yaw角#################
        project_vehicle['yaw_rad'] = self._calYaw(projection_axis[0],projection_axis[1])
        #################投影速度#################
        v_vector = self._calProjectVector([vehicle['v_mps']*math.cos(vehicle['yaw_rad']),vehicle['v_mps']*math.sin(vehicle['yaw_rad'])],
                                            projection_axis)
        project_vehicle['v_mps'] = math.sqrt(v_vector[0]**2+v_vector[1]**2)
        #################投影加速度#################
        a_vector = self._calProjectVector([vehicle['acc_mpss']*math.cos(vehicle['yaw_rad']),vehicle['acc_mpss']*math.sin(vehicle['yaw_rad'])],
                                            projection_axis)
        project_vehicle['acc_mpss'] = math.sqrt(a_vector[0]**2+a_vector[1]**2)

        self._projection_vehicle_S.append(project_vehicle)

    
    def _calYaw(self,dx,dy):
        """计算yaw角.
        输入:dx,dy——向量
        输出:theta——yaw角
        """
        theta = math.atan(dy/dx)
        if dx<0:
            theta += 3.1415926
        if theta<0:
            theta += 3.1415926*2
        return theta
     
    
    def _calProjectVector(self,vec1,vec2):
        """计算vec1车在vec2车上的投影
        输入:vec1——投影向量,vec2——投影轴
        输出:vec3——结果向量
        """
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] #点乘结果
        dot = dot/(vec2[0]**2 + vec2[1]**2)
        
        # Convert vec2 to a numpy array for element-wise multiplication
        vec2_np = np.array(vec2)
        
        return vec2_np*dot

    
    def __collisionDetect(self,v1,v2):
        """两辆车碰撞检测."""
        #计算车辆八个顶点
        x=v1[0]
        y=v1[1]
        yaw=v1[2]
        width=v1[3]+0.2
        length=v1[4]+0.2
        v1_p=[self.__vehicle2Global([x,y],[length/2,width/2],yaw),
              self.__vehicle2Global([x,y],[length/2,-width/2],yaw),
              self.__vehicle2Global([x,y],[-length/2,-width/2],yaw),
              self.__vehicle2Global([x,y],[-length/2,width/2],yaw)]
        x=v2[0]
        y=v2[1]
        yaw=v2[2]
        width=v2[3]+0.2
        length=v2[4]+0.2
        v2_p=[self.__vehicle2Global([x,y],[length/2,width/2],yaw),
              self.__vehicle2Global([x,y],[length/2,-width/2],yaw),
              self.__vehicle2Global([x,y],[-length/2,-width/2],yaw),
              self.__vehicle2Global([x,y],[-length/2,width/2],yaw)]
        #投影1
        width=v1[3]+0.2
        length=v1[4]+0.2
        axis=[v1_p[0][0]-v1_p[1][0],v1_p[0][1]-v1_p[1][1]]
        vec=[v2_p[0][0]-v1_p[1][0],v2_p[0][1]-v1_p[1][1]]
        projection_dis1=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[1][0]-v1_p[1][0],v2_p[1][1]-v1_p[1][1]]
        projection_dis2=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[2][0]-v1_p[1][0],v2_p[2][1]-v1_p[1][1]]
        projection_dis3=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[3][0]-v1_p[1][0],v2_p[3][1]-v1_p[1][1]]
        projection_dis4=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        min_dis=min(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        max_dis=max(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        if min_dis>width or max_dis<0:
            return False
        #投影2
        axis=[v1_p[2][0]-v1_p[1][0],v1_p[2][1]-v1_p[1][1]]
        vec=[v2_p[0][0]-v1_p[1][0],v2_p[0][1]-v1_p[1][1]]
        projection_dis1=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[1][0]-v1_p[1][0],v2_p[1][1]-v1_p[1][1]]
        projection_dis2=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[2][0]-v1_p[1][0],v2_p[2][1]-v1_p[1][1]]
        projection_dis3=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v2_p[3][0]-v1_p[1][0],v2_p[3][1]-v1_p[1][1]]
        projection_dis4=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        min_dis=min(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        max_dis=max(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        if min_dis>length or max_dis<0:
            return False
        #投影3
        width=v2[3]+0.2
        length=v2[4]+0.2
        axis=[v2_p[2][0]-v2_p[1][0],v2_p[2][1]-v2_p[1][1]]
        vec=[v1_p[0][0]-v2_p[1][0],v1_p[0][1]-v2_p[1][1]]
        projection_dis1=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[1][0]-v2_p[1][0],v1_p[1][1]-v2_p[1][1]]
        projection_dis2=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[2][0]-v2_p[1][0],v1_p[2][1]-v2_p[1][1]]
        projection_dis3=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[3][0]-v2_p[1][0],v1_p[3][1]-v2_p[1][1]]
        projection_dis4=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        min_dis=min(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        max_dis=max(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        if min_dis>length or max_dis<0:
            return False
        #投影4
        axis=[v2_p[0][0]-v2_p[1][0],v2_p[0][1]-v2_p[1][1]]
        vec=[v1_p[0][0]-v2_p[1][0],v1_p[0][1]-v2_p[1][1]]
        projection_dis1=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[1][0]-v2_p[1][0],v1_p[1][1]-v2_p[1][1]]
        projection_dis2=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[2][0]-v2_p[1][0],v1_p[2][1]-v2_p[1][1]]
        projection_dis3=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        vec=[v1_p[3][0]-v2_p[1][0],v1_p[3][1]-v2_p[1][1]]
        projection_dis4=(axis[0]*vec[0]+axis[1]*vec[1])/math.hypot(axis[0],axis[1])
        min_dis=min(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        max_dis=max(projection_dis1,projection_dis2,projection_dis3,projection_dis4)
        if min_dis>width or max_dis<0:
            return False
        return True
    
    
    def __vehicle2Global(self,v,point,yaw):
        x1=point[0]
        y1=point[1]
        x2=x1*math.cos(yaw)-y1*math.sin(yaw)+v[0]
        y2=x1*math.sin(yaw)+y1*math.cos(yaw)+v[1]
        return [x2,y2]
