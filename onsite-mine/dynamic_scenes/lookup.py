from enum import Enum, unique
import math
import time


@unique
class VehicleType(Enum):
    MineTruck_XG90G = 1  # 车辆类型:徐工宽体车90
    MineTruck_NTE200 = 2  # 车辆类型:北方股份电动论矿卡NTE200


class MineTruckXG90G:
    def __init__(self):
        # 车辆最大外形轮廓
        self.length = 9.0
        self.width = 4.0
        self.locationPoint2Head = 6.5
        self.locationPoint2Rear = 2.5
        # 道路边界碰撞检测框轮廓
        self.collision_l = 9.0
        self.collision_w = 4.0
        self.collision_lrear = 6.5
        self.collision_lhead = 2.5


class MineTruckNTE200:
    def __init__(self):
        # 车辆最大外形轮廓
        self.length = 13.0
        self.width = 6.7
        self.locationPoint2Head = 9.2
        self.locationPoint2Rear = 3.8
        # 道路边界碰撞检测框轮廓
        self.collision_l = 8.8
        self.collision_w = 5.5
        self.collision_lrear = 1.6
        self.collision_lhead = 7.2


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Point2D:
    """2D平面的位姿点."""

    def __init__(self, x, y, yaw_rad: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw_rad


class CollisionLookup:
    """道路边界碰撞检测.
       通过建立一个查找表(lookup table)，能够快速检测给定位置和方向的车辆是否会与环境中的障碍物边界发生碰撞.
       注：与其它车辆进行碰撞检测不按照该方法.
    ## 1.碰撞查找表的生成过程
        首先，根据车辆尺寸和预设的参数，初始化车辆在不同位置和方向上的可能状态。
        对于每个可能的位置和方向，计算车辆四个角的位置，并根据这些角的位置来填充一个称为C空间的二维数组。C空间反映了车辆所占的空间。
        然后，利用直线扫描算法（Bresenham算法的一个变种），标记出所有可能与车辆发生碰撞的栅格。
        最后，将这些数据保存到查找表中，供碰撞检测时查询使用。
    ## 2.碰撞检测过程
        给定一个车辆位姿(x，y,yaw) 和地图，函数首先将这些参数转换为查找表中的索引。
        然后，遍历对应的查找表项，检查车辆所占的每个栅格是否与地图上的障碍物重叠。
        如果存在重叠（即发现碰撞），则立即返回碰撞标志。
    ## 3.性能优化
        这个系统的设计允许快速碰撞检测，避免了复杂计算和直接的几何碰撞检测算法。
        通过预计算和存储车辆在不同位置和方向上的占用格，显著减少了实时计算的需要，适用于需要快速响应的自动驾驶和机器人导航系统。
    """

    def __init__(self, type: VehicleType = VehicleType.MineTruck_XG90G):
        """构造函数初始化一系列参数，如车辆尺寸、碰撞检测精度、方向的离散化等，并计算出用于碰撞检测的查找表."""
        # !初始化车辆参数
        if type == VehicleType.MineTruck_NTE200:
            self.vehicle = MineTruckXG90G()
        elif type == VehicleType.MineTruck_XG90G:
            self.vehicle = MineTruckXG90G()
        else:
            self.vehicle = MineTruckXG90G()

        # !初始化碰撞检测精度参数
        self._headings = 72  # 空间分辨率,方向的离散化程度,即在全圆中考虑的不同方向的数量.
        self._c_size = 0.2  # 碰撞检测的空间分辨率,单位是米.
        self._position_resolution = 1  # _c_size*_c_size 格子进一步细化
        self._positions = self._position_resolution**2
        self.expansion_dis = 0.0  # 碰撞尺寸,单位米;
        # 基于车辆尺寸和self._c_size参数计算的边界框大小，用于定义碰撞检测时考虑的空间区域
        self._bbSize = math.ceil((math.sqrt(self.vehicle.collision_w**2 + (self.vehicle.collision_l * 2) ** 2) + self.expansion_dis) / self._c_size)
        self._delt_heading_rad = 3.1415926 * 2 / self._headings
        self._delt_heading_rad = 3.1415926 * 2 / self._headings

        # 生成每个栅格的离散点位
        self._points = [Point2D(0.0, 0.0) for i in range(self._positions)]
        for i in range(self._position_resolution):
            for j in range(self._position_resolution):
                pt = Point2D(1.0 / self._position_resolution * j, 1.0 / self._position_resolution * i)
                self._points[i * self._position_resolution + j] = pt
        self._c = Point2D(0.0, 0.0)  # 车辆中心
        self._p = [Point2D(0.0, 0.0) for i in range(4)]  # 车辆的四个角
        self._np = [Point2D(0.0, 0.0) for i in range(4)]  # 旋转后车辆的四个角
        self._lookup = [list() for i in range(self._positions * self._headings)]  # 表格

        for q in range(self._positions):
            theta = 0.0
            # 计算车辆中心
            self._c.x = self._bbSize / 2.0 + self._points[q].x
            self._c.y = self._bbSize / 2.0 + self._points[q].y
            # 计算车辆四个角位置
            self._p[0].x = self._c.x - self.vehicle.collision_lrear / self._c_size
            self._p[0].y = self._c.y - self.vehicle.collision_w / 2 / self._c_size
            self._p[1].x = self._c.x - self.vehicle.collision_lrear / 2 / self._c_size
            self._p[1].y = self._c.y + self.vehicle.collision_w / 2 / self._c_size
            self._p[2].x = self._c.x + self.vehicle.collision_lhead / self._c_size
            self._p[2].y = self._c.y + self.vehicle.collision_w / 2 / self._c_size
            self._p[3].x = self._c.x + self.vehicle.collision_lhead / self._c_size
            self._p[3].y = self._c.y - self.vehicle.collision_w / 2 / self._c_size

            for o in range(self._headings):
                self._cSpace = [False for i in range(self._bbSize**2)]  # 初始化C空间

                # 旋转变换
                for i in range(4):
                    temp_x = self._p[i].x - self._c.x
                    temp_y = self._p[i].y - self._c.y
                    self._np[i].x = temp_x * math.cos(theta) - temp_y * math.sin(theta) + self._c.x
                    self._np[i].y = temp_x * math.sin(theta) + temp_y * math.cos(theta) + self._c.y
                theta += self._delt_heading_rad

                # 单元格顺时针遍历
                for i in range(4):
                    start = Point2D(0, 0)
                    end = Point2D(0, 0)
                    # 起始点和终点设置
                    if i < 3:
                        start = self._np[i]
                        end = self._np[i + 1]
                    else:
                        start = self._np[i]
                        end = self._np[0]

                    # x,y向下取整得到下标
                    x_id = math.floor(start.x)
                    y_id = math.floor(start.y)
                    self._cSpace[y_id * self._bbSize + x_id] = True  # 占用

                    # 搜索起点到终点的方向向量
                    t = Point2D(end.x - start.x, end.y - start.y)
                    # 下一步移动方向(按格点移动)
                    step_x = sign(t.x)
                    step_y = sign(t.y)
                    # 下一步移动距离占总移动距离的比例(x,y方向)
                    t_delta_x = 1000
                    t_delta_y = 1000
                    if t.x != 0:
                        t_delta_x = 1.0 / abs(t.x)  # 一个格点占x移动距离的比例
                    if t.y != 0:
                        t_delta_y = 1.0 / abs(t.y)  # 一个格点占y移动距离的比例
                    t_max_x, t_max_y = 0.0, 0.0
                    if step_x > 0:
                        t_max_x = t_delta_x * (1 - (start.x - math.trunc(start.x)))
                    else:
                        t_max_x = t_delta_x * (start.x - math.trunc(start.x))
                    if step_y > 0:
                        t_max_y = t_delta_y * (1 - (start.y - math.trunc(start.y)))
                    else:
                        t_max_y = t_delta_y * (start.y - math.trunc(start.y))

                    # 边界占用栅格搜索
                    while math.floor(end.x) != x_id or math.floor(end.y) != y_id:
                        # 当x所占比例小,且移动后距离终点更近时,移动x
                        if t_max_x < t_max_y and abs(x_id + step_x - math.floor(end.x)) < abs(x_id - math.floor(end.x)):
                            t_max_x += t_delta_x
                            x_id += step_x
                            self._cSpace[y_id * self._bbSize + x_id] = True
                        elif t_max_y < t_max_x and abs(y_id + step_y - math.floor(end.y)) < abs(y_id - math.floor(end.y)):
                            t_max_y += t_delta_y
                            y_id += step_y
                            self._cSpace[y_id * self._bbSize + x_id] = True
                        elif 2 >= abs(x_id - math.floor(end.x)) + abs(y_id - math.floor(end.y)):
                            if abs(x_id - math.floor(end.x)) > abs(y_id - math.floor(end.y)):
                                x_id += step_x
                                self._cSpace[y_id * self._bbSize + x_id] = True
                            else:
                                y_id += step_y
                                self._cSpace[y_id * self._bbSize + x_id] = True
                        else:
                            print("\n--->tie occured,please check for error in script\n")
                            break

                # 填满图形
                hcross1, hcross2 = 0, 0
                for i in range(self._bbSize):
                    inside = False
                    # 寻找覆盖区域的起点和终点
                    for k in range(self._bbSize):
                        if self._cSpace[i * self._bbSize + k] == True and inside == False:
                            hcross1 = k
                            inside = True
                        elif self._cSpace[i * self._bbSize + k] == True and inside == True:
                            hcross2 = k
                    # 如果没有找到覆盖区域
                    if inside == False:
                        continue
                    # 填充覆盖区域
                    for j in range(self._bbSize):
                        if j >= hcross1 and j <= hcross2:
                            self._cSpace[i * self._bbSize + j] = True

                # 设置表格
                for i in range(self._bbSize):
                    for j in range(self._bbSize):
                        if self._cSpace[i * self._bbSize + j]:
                            self._lookup[q * self._headings + o].append(Point2D(j - math.floor(self._c.x), i - math.floor(self._c.y)))

    def collision_detection(self, x: float, y: float, yaw: float, image):
        """接收位置(x，y)、偏航角和一个二维地图，判断在该位置和方向上的车辆是否会碰撞.

        Args:
            x (float): x,y——与地图原点的相对坐标
            y (float): x,y——与地图原点的相对坐标
            yaw (float): 车辆偏航角
            image (np.array): 二值化地图(表示是否被占用)

        Returns:
            bool: 是否与边界发生碰撞.
        """
        start_time = time.perf_counter()

        x = x / 0.1  # self._c_size
        y = y / 0.1  # self._c_size
        X = math.floor(x)
        Y = math.floor(y)
        iX = math.floor((x - X) * self._position_resolution)
        iY = math.floor((y - Y) * self._position_resolution)
        while yaw > math.pi * 2:
            yaw -= math.pi * 2
        while yaw < 0:
            yaw += math.pi * 2
        iH = math.floor(yaw / self._delt_heading_rad)
        idx = iY * self._position_resolution * self._headings + iX * self._headings + iH

        for i in range(len(self._lookup[idx])):
            cX = X + self._lookup[idx][i].x * 2
            cY = Y + self._lookup[idx][i].y * 2
            if image[cY][cX] == False:
                return True
        end_time = time.perf_counter()
        # print("##log##车辆与道路边界碰撞检测时间:"+str(end_time-start_time))
        return False
