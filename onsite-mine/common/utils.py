# 内置库 
import math

# 第三方库
import numpy as np
import matplotlib.pyplot as plt
from typing import List,Dict,Set,Tuple,Union



def filterForYawBySinAndCos(angle_rad,Q_oneDim,R_oneDim,smooth_N1,smooth_N2):
    """ 对航向值的 sin cos 值 进行过滤
        
        Parameters
        ----------
        angle_rad :角度列表 ,单位 rad  # 列向量
        Q_oneDim :xxxx   
        R_oneDim :xxxx    
        smooth_N1 :xxxx 
        smooth_N2 :xxxx 

        Returns
        -------
        angle_rad_KF_smooth : 返回 滤波后的 angle ,单位 rad          
    """
    NTemp = angle_rad.shape[0]
    angle_sin = np.sin(angle_rad)
    angle_cos = np.cos(angle_rad)
    angle_rad_KF_smooth = np.zeros(NTemp)
    
    angle_sin_KF = KalmanFilter_FirstOrder(angle_sin,Q_oneDim,R_oneDim)
    angle_sin_KF_smooth = angle_sin_KF #不再进行移动平均滤波

    angle_cos_KF = KalmanFilter_FirstOrder(angle_cos,Q_oneDim,R_oneDim)
    angle_cos_KF_smooth = angle_cos_KF

    for i in range(NTemp):
        if angle_sin_KF_smooth[i] >= 0 and angle_cos_KF_smooth[i] >= 0: # 第一象限
            angle_rad_KF_smooth[i] = np.arccos(angle_cos_KF_smooth[i])
        elif angle_sin_KF_smooth[i] >= 0 and angle_cos_KF_smooth[i] < 0: # 第二象限
            angle_rad_KF_smooth[i] = np.arccos(angle_cos_KF_smooth[i])
        elif angle_sin_KF_smooth[i] < 0 and angle_cos_KF_smooth[i] < 0: # 第三象限
            angle_rad_KF_smooth[i] = 2* np.pi - np.arccos(angle_cos_KF_smooth[i])
        elif angle_sin_KF_smooth[i] < 0 and angle_cos_KF_smooth[i] >= 0: # 第四象限
            angle_rad_KF_smooth[i] = 2* np.pi - np.arccos(angle_cos_KF_smooth[i])
        else:
            # print("### log ### 一共 xxxxx")
            pass

    return angle_rad_KF_smooth


def KalmanFilter_FirstOrder(data,Q,R):
    """ 线性卡尔曼滤波,一维
    """
    N = len(data)
    KalmanGain = np.zeros(N)
    X_hat = np.zeros(N)
    X_hat_pre = np.zeros(N)
    P_covariance = np.zeros(N)
    P_covariance_pre = np.zeros(N)

    X_hat_pre[0] = data[0]
    P_covariance[0] = 1

    # 系统矩阵
    A = 1  # 状态转移矩阵
    H = 1  # 测量矩阵
    I = np.eye(1)

    for k in range(N-1):
        # 卡尔曼滤波过程
        X_hat_pre[k+1] = A * X_hat[k]  # 1) 预测:先验状态估计值
        P_covariance_pre[k+1] = A * P_covariance[k] * A + Q  # 2) 预测:先验预测协方差矩阵

        KalmanGain[k+1] = P_covariance_pre[k+1] * H /(H * P_covariance_pre[k+1] * H + R)  # 3) 校正:计算Kalman增益矩阵

        X_hat[k+1] = X_hat_pre[k+1] + KalmanGain[k+1] * (data[k+1] - H * X_hat_pre[k+1])  # 4) 校正:后验状态估计更新
        P_covariance[k+1] = (I - KalmanGain[k+1] * H) * P_covariance_pre[k+1]  # 5) 校正:误差协方差矩阵更新

    return X_hat


def KalmanFilter_linear(measurements,initial_state,
                        transition_matrix,observation_matrix,
                        process_noise_covariance,measurement_noise_covariance):
    """ 有问题,未调好
    """
    num_measurements = len(measurements)
    # num_states = initial_state.shape[0]
    num_states = 1

    # 初始化状态估计和协方差矩阵
    state_estimation = np.zeros((num_states,num_measurements))
    state_covariance = np.zeros((num_states,num_states,num_measurements))
    
    # 初始化先验状态估计和协方差矩阵
    predicted_state_estimation = np.zeros((num_states,num_measurements))
    predicted_state_covariance = np.zeros((num_states,num_states,num_measurements))

    # 初始状态估计
    state_estimation[:,0] = initial_state
    state_covariance[:,:,0] = np.eye(num_states)

    for t in range(1,num_measurements):
        # 预测步骤
        predicted_state_estimation[:,t] = transition_matrix @ state_estimation[:,t-1]
        predicted_state_covariance[:,:,t] = transition_matrix @ state_covariance[:,:,t-1] @ transition_matrix.T + process_noise_covariance
        
        # 更新步骤
        kalman_gain = predicted_state_covariance[:,:,t] @ observation_matrix.T @ np.linalg.inv(observation_matrix @ \
            predicted_state_covariance[:,:,t] @ observation_matrix.T + measurement_noise_covariance)
        state_estimation[:,t] = predicted_state_estimation[:,t] + kalman_gain @ (measurements[t] - observation_matrix @ \
            predicted_state_estimation[:,t])
        state_covariance[:,:,t] = (np.eye(num_states) - kalman_gain @ observation_matrix) @ predicted_state_covariance[:,:,t]
    
    return state_estimation


def smooth_data(data,window_size):
    """This function applies a simple moving average filter to the data by convolving it with a window of ones.
        该函数通过将数据与一个1的窗口进行卷积,对数据应用一个简单的移动平均过滤器.
    """
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data,window,mode='valid')
   
    return smoothed_data


def surrounding_id (observation,ego_x,ego_y,r):
    '''该函数输入车辆的XY坐标以及检查范围,输出车辆范围内的其他车辆ID
    '''
    id_veh = []
    for key,value in observation['vehicle_info'].items():
        if key != 'ego':
            dis = math.hypot(abs(observation['vehicle_info'][key]['x']-ego_x),abs(observation['vehicle_info'][key]['y']-ego_y))
            if dis < r:
                id_veh.append(key)
    return id_veh


# ! 以上暂时无用

      
def find_nearest_point_index(X:float,Y:float,path_points_XY:np.array) ->int:
    """find id of nearest a path point

    Args:
        X (float):匹配点x坐标;
        Y (float):匹配点y坐标;
        path_points_XY (np.array):路径点序列;保证前两列为x y坐标值即可

    Returns:
        int:最近的路径点id
    """
    min_distance = math.inf
    nearest_id = -1
    for i,point in enumerate(path_points_XY):
        point_x,point_y,_ = point
        distance = math.sqrt((X - point_x) ** 2 + (Y - point_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_id = i
    return nearest_id
    
    
def find_preview_point_index(X:float,Y:float,preview_dis:float,path_points_XY:np.array) ->int:
    """按照预瞄距离preview_dis查找参考路径点索引

    Args:
        X (float):匹配点x坐标
        Y (float):匹配点y坐标
        preview_dis (float):向前方向参考路径预瞄距离
        path_points_XY (np.array):路径点序列;保证前两列为x y坐标值即可

    Returns:
        int:预瞄路径点的id
    """
    nearest_id = find_nearest_point_index(X,Y,path_points_XY)
    preview_id = nearest_id
    accumulated_distance = 0.0
    for i in range(nearest_id,len(path_points_XY-1)):
        distance = math.sqrt((path_points_XY[i+1][0] - path_points_XY[i][0])**2 + (path_points_XY[i+1][1]- path_points_XY[i][1])**2)
        accumulated_distance += distance  # 使用累加的距离
        if accumulated_distance > preview_dis:   
            break #如果预瞄距离到达最后id,就给出最后的id
        preview_id = i+1
        
    return preview_id


def find_preview_point_index_over_end(X:float,Y:float,preview_dis:float,path_points_XY:np.array) -> Union[int,bool]:
    """按照预瞄距离preview_dis查找参考路径点索引,允许超出界限end一定程度

    Args:
        X (float):匹配点x坐标
        Y (float):匹配点y坐标
        preview_dis (float):向前方向参考路径预瞄距离
        path_points_XY (np.array):路径点序列;保证前两列为x y坐标值即可

    Returns:
        int:预瞄路径点的id
    """
    over_index_distance_ratio = 0.2 # !索引超界;设置预瞄距离可以超过 20% [可配置参数项]
    flag_preview_dis_over_end = False #初始值
    
    nearest_id = find_nearest_point_index(X,Y,path_points_XY)
    preview_id = nearest_id
    accumulated_distance = 0.0
    for i in range(nearest_id,len(path_points_XY-1)):
        if i+1 >= len(path_points_XY)-1:###### 索引超界判断,并处理 #####
            over_index_distance_ratio_temp=(preview_dis - accumulated_distance)/accumulated_distance
            if over_index_distance_ratio_temp<over_index_distance_ratio:
                flag_preview_dis_over_end = False
                preview_id = i
                break
            else:
                flag_preview_dis_over_end = True
                preview_id = i
                break
            ###### 索引超界判断,并处理 end#####
        else:
            distance = math.sqrt((path_points_XY[i+1][0] - path_points_XY[i][0])**2 + (path_points_XY[i+1][1]- path_points_XY[i][1])**2)
            accumulated_distance += distance  # 使用累加的距离
            if accumulated_distance > preview_dis:   
                break #如果预瞄距离到达最后id,就给出最后的id
            preview_id = i+1
        
    return preview_id,flag_preview_dis_over_end


def compute_two_pose_error(X1:float,Y1:float,Yaw1:float,X2:float,Y2:float,Yaw2:float)-> Tuple[float,float,float]:
    """计算当前位姿点 pose1 = (X1,Y1,Yaw1) 到目标位姿点 pose2 = (X2,Y2,Yaw2) 的误差项(位姿关系)

    Args:
        X1 (float):单位m
        Y1 (float):单位m
        Yaw1 (float):单位 rad , 东偏北,范围(0,2*pi]
        X2 (float):单位m
        Y2 (float):单位m
        Yaw2 (float):单位 rad , 东偏北,范围(0,2*pi]

    Returns:
        Tuple[float,float,float]:lateral_error 横向误差(左正右负),pose1在pose2左侧为正;
                                    longitudinal_error 纵向误差(前正后负),pose1在pose2前方为正;
                                    heading_error_rad 航向误差(顺正逆负) ,pose1相对于pose2顺时针旋转为正;
    """
    dx = X1 - X2
    dy = Y1 - Y2

    lateral_error = math.cos(Yaw2) * dy - math.sin(Yaw2) * dx
    longitudinal_error = math.sin(Yaw2) * dy + math.cos(Yaw2) * dx

    heading_error_rad_temp = Yaw1 - Yaw2 # Calculate heading error in radians
    # Normalize heading error to the range [-pi,pi]
    heading_error_rad = (heading_error_rad_temp + np.pi) % (2 * np.pi) - np.pi

    return lateral_error,longitudinal_error,heading_error_rad


def calculate_longitudinal_distance(point,dubinspose):
    """计算当前车辆定位点 point = (x2,y2) 到位姿点 dubinspose = (x1,y1,yaw1) 的纵向距离,"""
    # ! 未验证
    x1,y1,yaw1 = dubinspose
    x2,y2 = point
    
    # 计算法线方向向量
    normal_vector = (math.cos(yaw1 + math.pi/2),math.sin(yaw1 + math.pi/2))
    
    # 计算向量差
    vector_diff = (x2 - x1,y2 - y1)
    
    # 计算纵向距离
    distance = vector_diff[0] * normal_vector[0] + vector_diff[1] * normal_vector[1]
    
    return distance
 
 
def calculate_vehicle_corners(length,width,locationPoint2Head,locationPoint2Rear,x,y,yaw_rad):
    """根据提供的车辆形状参数和位置信息,计算车辆的四个顶点坐标. 

    Args:
        length (_type_): 车辆长度
        width (_type_): 车辆宽度
        locationPoint2Head (_type_): 车辆定位中心距离车头长度
        locationPoint2Rear (_type_): 车辆定位中心距离车尾部长度
        x (_type_): 车辆定位点坐标x
        y (_type_): 车辆定位点坐标y
        yaw_rad (_type_): 车辆航向 

    Returns:
        float: 车辆四个角点坐标(x,y).
    """
    
    # 计算车辆中心点
    center = np.array([x,y])
    
    # 计算方向矢量
    direction = np.array([np.cos(yaw_rad),np.sin(yaw_rad)])
    perpendicular_direction = np.array([-np.sin(yaw_rad),np.cos(yaw_rad)])
    
    # 计算车辆前后中心点
    front_center = center + locationPoint2Head * direction
    rear_center = center - locationPoint2Rear * direction
    
    # 计算车辆半长和半宽矢量
    half_length_vec = 0.5 * length * direction
    half_width_vec = 0.5 * width * perpendicular_direction
    
    # 计算车辆四个角点
    front_left_corner = front_center + half_width_vec
    front_right_corner = front_center - half_width_vec
    rear_left_corner = rear_center + half_width_vec
    rear_right_corner = rear_center - half_width_vec

    return front_left_corner.tolist(),front_right_corner.tolist(),rear_left_corner.tolist(),rear_right_corner.tolist()


def is_inside_polygon(x:float,y:float,polygon:Dict[str,list]) -> bool:
    """使用射线法判断点 point(x,y) 是否在多边形内部.
    Args:
        x (float):x坐标
        y (float):y坐标
        polygon (Dict[str,list]):描述多边形的字典,例如:
            {'x':[895.3440941,890.0361717,907.9749535,913.2828758],
             'y':[2314.994762,2325.517345,2334.566173,2324.04359]}
    Returns:
        bool:如果点在多边形内部返回True,否则返回False
    """
    # 初始化交点数量为0
    intersect_count = 0
    
    num_points = len(polygon['x'])
    for i in range(num_points):
        # 为每一个线段提取开始点p1和结束点p2
        p1 = (polygon['x'][i],polygon['y'][i])
        p2 = (polygon['x'][(i + 1) % num_points],polygon['y'][(i + 1) % num_points])  # 取%保证可以回到顶点

        # 确保 p1 的 y 值小于 p2 的 y 值
        if p1[1] > p2[1]:
            p1,p2 = p2,p1
        
        # 判断水平射线是否与线段有交点
        if p1[1] <= y < p2[1]:
            x_intersect = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]) + p1[0]
            if x < x_intersect:
                intersect_count += 1
    
    # 如果交点数量为奇数,返回True,否则返回False
    return intersect_count % 2 == 1


 
class PolynomialInterpolation:
    """笛卡尔坐标系下五次多项式插值."""
    def __init__(self,pt1,pt2,v1,v2,a1,a2,delta_t=0.1):
        self.pt1 = pt1  # 起始点位置
        self.pt2 = pt2  # 终止点位置
        self.v1 = v1    # 起始点速度
        self.v2 = v2    # 终止点速度
        self.a1 = a1    # 起始点加速度
        self.a2 = a2    # 终止点加速度
        self.delta_t = delta_t  # 时间步长


    def quintic_polynomial_interpolation(self,t0=0,t1=5.0) ->np.array:
        """Quintic polynomial curve interpolation   五次多项式
            起点时间 t0 =0; 终点时间t1 =5 #多项式曲线固定时长 5s
        """
        # t0 = 0
        # t1 = 5
        # T = np.array([[t0**5,  t0**4,   t0**3,   t0**2,t0,  1],
        #               [5*t0**4,4*t0**3, t0**2,   2*t0,1,   0],
        #               [20*t0**3,12*t0**2,6*t0,    2,   0,   0],
        #               [t1**5,  t1**4,   t1**3,   t1**2,t1,  1],
        #               [5*t1**4,4*t1**3, 3*t1**2, 2*t1,1,   0],
        #               [20*t1**3,12*t1**2,6*t1,    2,   0,   0]])
        T=np.array([[0,      0,       0,       0,       0,   1],
                    [0,      0,       0,       0,       1,   0],
                    [0,      0,       0,       2,       0,   0],
                    [t1**5,  t1**4,   t1**3,   t1**2,   t1,  1],
                    [5*t1**4,4*t1**3, 3*t1**2, 2*t1,    1,   0],
                    [20*t1**3,12*t1**2,6*t1,    2,       0,   0]]) #T矩阵
        T_inv=np.linalg.inv(T)
        X = np.array([[self.pt1[0]],[self.v1[0]],[self.a1[0]],[self.pt2[0]],[self.v2[0]],[self.a2[0]]]) #X矩阵
        Y = np.array([[self.pt1[1]],[self.v1[1]],[self.a1[1]],[self.pt2[1]],[self.v2[1]],[self.a2[1]]]) #Y矩阵
        A = np.dot(T_inv,X)
        B = np.dot(T_inv,Y)

        # num = int(t1/self.delta_t)
        # 将时间从t0到t1离散化,获得离散时刻的轨迹坐标
        t = np.arange(t0,t1+self.delta_t,self.delta_t)
        interpolation_trajectory = np.zeros((len(t),4))  # 1-4列分别存放x,y,vx,vy
        for i in range(len(t)):
            # x位置坐标
            interpolation_trajectory[i,0] = np.dot([t[i]**5,t[i]**4,t[i]**3,t[i]**2,t[i],1],A)
            # y位置坐标
            interpolation_trajectory[i,1] = np.dot([t[i]**5,t[i]**4,t[i]**3,t[i]**2,t[i],1],B)
            # x速度
            interpolation_trajectory[i,2] = np.dot([5*t[i]**4,4*t[i]**3,3*t[i]**2,2*t[i],1,0],A)
            # y速度
            interpolation_trajectory[i,3] = np.dot([5*t[i]**4,4*t[i]**3,3*t[i]**2,2*t[i],1,0],B)

        return interpolation_trajectory


    def quintic_polynomial_interpolation_2(self,t1=5.0):
        """Quintic polynomial curve interpolation   五次多项式
            起点时间 t0 =0; 终点时间t1 =5 #多项式曲线固定时长 5s
            @吴佳琪 该部分有问题;
        """
        # t0 = 0
        # t1 = 5
        # T = np.array([[t0**5,  t0**4,   t0**3,   t0**2,t0,  1],
        #               [5*t0**4,4*t0**3, t0**2,   2*t0,1,   0],
        #               [20*t0**3,12*t0**2,6*t0,    2,   0,   0],
        #               [t1**5,  t1**4,   t1**3,   t1**2,t1,  1],
        #               [5*t1**4,4*t1**3, 3*t1**2, 2*t1,1,   0],
        #               [20*t1**3,12*t1**2,6*t1,    2,   0,   0]])
        T=np.array([[0,      0,       0,       0,       0,   1],
                    [0,      0,       0,       0,       1,   0],
                    [0,      0,       0,       2,       0,   0],
                    [t1**5,  t1**4,   t1**3,   t1**2,   t1,  1],
                    [5*t1**4,4*t1**3, 3*t1**2, 2*t1,    1,   0],
                    [20*t1**3,12*t1**2,6*t1,    2,       0,   0]]) #T矩阵
        T_inv=np.linalg.inv(T)
        X = np.array([[self.pt1[0]],[self.v1[0]],[self.a1[0]],[self.pt2[0]],[self.v2[0]],[self.a2[0]]]) #X矩阵
        Y = np.array([[self.pt1[1]],[self.v1[1]],[self.a1[1]],[self.pt2[1]],[self.v2[1]],[self.a2[1]]]) #Y矩阵
        A = np.dot(T_inv,X)
        B = np.dot(T_inv,Y)

        interpolation_trajectory = []
        num = int(t1/self.delta_t)
        for i in range(1,num+1):#1,2,3,...,50
            dt = i * self.delta_t
            T_dt = np.array([[0,      0,       0,       0,       0,   1],
                             [0,      0,       0,       0,       1,   0],
                             [0,      0,       0,       2,       0,   0],
                             [dt**5,  dt**4,   dt**3,   dt**2,   dt,  1],
                             [5*dt**4,4*dt**3, 3*dt**2, 2*dt,    1,   0],
                             [20*dt**3,12*dt**2,6*dt,    2,       0,   0]]) #T矩阵
            X_dt = np.dot(T_dt,X)
            Y_dt = np.dot(T_dt,Y)
            x_dt = X_dt[3][0]
            y_dt = Y_dt[3][0]
            v_dt = X_dt[4][0]
            point = [x_dt,y_dt,v_dt]
            interpolation_trajectory.append(point)

        return interpolation_trajectory


    # TODO 三次多项式曲线生成函数
    def cubic_polynomial_interpolation_2(self,t1=5.0):
        pass
    


if __name__ == "__main__":
    """测试代码"""
    pt1 = [0,-1.75]    # 起始点位置
    pt2 = [20,1.75]   # 终止点位置
    v1 = [5,0]     # 起始点速度
    v2 = [5,0]     # 终止点速度
    a1 = [0,0]     # 起始点加速度
    a2 = [0,0]     # 终止点加速度

    interpolator = PolynomialInterpolation(pt1,pt2,v1,v2,a1,a2)
    interpolation_trajectory = interpolator.quintic_polynomial_interpolation()

    # 提取插值结果的 x 和 y 坐标
    x_values = [point[0] for point in interpolation_trajectory]
    y_values = [point[1] for point in interpolation_trajectory]

    # 绘制插值路径
    plt.plot(x_values,y_values,'r--',linewidth=1.5,marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quintic Polynomial Interpolation trajectory')
    plt.grid(True)
    plt.show()
    
    a=1 