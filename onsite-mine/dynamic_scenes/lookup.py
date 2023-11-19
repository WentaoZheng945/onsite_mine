import copy
import math
import time

def sign(x):
    if x>=0:return 1
    else:return -1

class Point2D:
    """2D平面的位姿点."""
    def __init__(self,x,y,yaw_rad:float=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw_rad

class CollisionLookup:
    def __init__(self):
        self._cSize = 0.2  # 每个单元格的大小
        self._position_resolution = 1  # 车辆位置分辨率
        self._positions = self._position_resolution**2  # 车辆位置的总数
        self._ego_width = 5.5  # 主车宽度
        self._ego_length = 8.8  # 主车长度
        self._locationPoint2Head = 7.2  # 后轴中心到前轮的长度
        self._locationPoint2Rear = 1.6  # 后轴中心到后轮的距离
        self._bbSize = math.ceil((math.sqrt(self._ego_width**2 + (self._locationPoint2Head*2)**2) + 4) / self._cSize)  # 包含车的栅格数
        self._headings = 72  # 72个离散角度
        self._delt_heading_rad = 3.1415926*2/72
        #生成每个栅格的离散点位
        self._points = [Point2D(0.,0.) for i in range(self._positions)]
        for i in range(self._position_resolution):
            for j in range(self._position_resolution):
                pt = Point2D(1.0/self._position_resolution*j,\
                                1.0/self._position_resolution*i)
                self._points[i*self._position_resolution+j]=pt  # 维度为1*self.resolution**2
        self._c = Point2D(0.,0.) # 车辆中心
        self._p = [Point2D(0.,0.) for i in range(4)] #车辆的四个角
        self._np = [Point2D(0.,0.) for i in range(4)] #旋转后车辆的四个角
        self._lookup = [list() for i in range(self._positions*self._headings)] #表格
        for q in range(self._positions):
            theta = 0.0
            #计算车辆中心
            self._c.x = self._bbSize / 2.0 + self._points[q].x
            self._c.y = self._bbSize / 2.0 + self._points[q].y
            #计算车辆四个角位置
            self._p[0].x = self._c.x - self._locationPoint2Rear/self._cSize  # 6
            self._p[0].y = self._c.y - self._ego_width/2/self._cSize  # 10.5
            self._p[1].x = self._c.x - self._locationPoint2Rear/2/self._cSize  # 6
            self._p[1].y = self._c.y + self._ego_width/2/self._cSize  # 17.5
            self._p[2].x = self._c.x + self._locationPoint2Head/self._cSize
            self._p[2].y = self._c.y + self._ego_width/2/self._cSize
            self._p[3].x = self._c.x + self._locationPoint2Head/self._cSize
            self._p[3].y = self._c.y - self._ego_width/2/self._cSize

            for o in range(self._headings):
                self._cSpace = [False for i in range(self._bbSize**2)] #初始化C空间

                #旋转变换
                for i in range(4):
                    temp_x = self._p[i].x - self._c.x
                    temp_y = self._p[i].y - self._c.y
                    self._np[i].x = temp_x*math.cos(theta) - temp_y*math.sin(theta) + self._c.x
                    self._np[i].y = temp_x*math.sin(theta) + temp_y*math.cos(theta) + self._c.y
                theta+=self._delt_heading_rad

                #单元格顺时针遍历
                for i in range(4):
                    start = Point2D(0,0)
                    end = Point2D(0,0)
                    #起始点和终点设置
                    if i<3:
                        start = self._np[i]  # (6, 10.5)
                        end = self._np[i+1]  # (6, 17.5)
                    else:
                        start = self._np[i]
                        end = self._np[0]
                    
                    #x,y向下取整得到下标
                    x_id = math.floor(start.x)  # 6
                    y_id = math.floor(start.y)  # 10
                    self._cSpace[y_id*self._bbSize+x_id] = True #占用

                    #搜索起点到终点的方向向量
                    t = Point2D(end.x - start.x, end.y - start.y)  # (0, 7)
                    #下一步移动方向(按格点移动)
                    step_x = sign(t.x)  # 1
                    step_y = sign(t.y)  # 1
                    #下一步移动距离占总移动距离的比例(x,y方向)
                    t_delta_x = 1000
                    t_delta_y = 1000
                    if t.x!=0:t_delta_x = 1./abs(t.x)  # 一个格点占x移动距离的比例
                    if t.y!=0:t_delta_y = 1./abs(t.y)  # 一个格点占y移动距离的比例
                    t_max_x,t_max_y = 0.,0.
                    if step_x > 0:t_max_x = t_delta_x*(1-(start.x - math.trunc(start.x)))
                    else:t_max_x = t_delta_x*(start.x - math.trunc(start.x))
                    if step_y > 0:t_max_y = t_delta_y*(1-(start.y - math.trunc(start.y)))
                    else:t_max_y = t_delta_y*(start.y - math.trunc(start.y))

                    #边界占用栅格搜索
                    # 这里的优先移动的原则为:优先移动距离较远的方向（即移动的一格相较于该方向的距离来说，微不足道（影响较小）的方向）
                    # 确定方向后，要再确定移动后在该方向上离目标点更近了
                    while math.floor(end.x)!=x_id or math.floor(end.y)!=y_id:
                        #当x所占比例小,且移动后距离终点更近时,移动x
                        if t_max_x<t_max_y and abs(x_id+step_x-math.floor(end.x))<abs(x_id-math.floor(end.x)):
                            t_max_x += t_delta_x
                            x_id += step_x
                            self._cSpace[y_id*self._bbSize+x_id]=True
                        elif t_max_y<t_max_x and abs(y_id+step_y-math.floor(end.y))<abs(y_id-math.floor(end.y)):
                            t_max_y += t_delta_y
                            y_id += step_y
                            self._cSpace[y_id*self._bbSize+x_id]=True
                        elif 2 >= abs(x_id-math.floor(end.x)) + abs(y_id-math.floor(end.y)):  # 判断终点
                            if abs(x_id-math.floor(end.x)) > abs(y_id-math.floor(end.y)):
                                x_id += step_x
                                self._cSpace[y_id*self._bbSize+x_id]=True
                            else:
                                y_id += step_y
                                self._cSpace[y_id*self._bbSize+x_id]=True
                        else:
                            print("\n--->tie occured,please check for error in script\n")
                            break

                #填满图形
                hcross1,hcross2=0,0
                for i in range(self._bbSize):
                    inside = False
                     #寻找覆盖区域的起点和终点
                    for k in range(self._bbSize):
                        if self._cSpace[i*self._bbSize+k]==True and inside==False:
                            hcross1 = k
                            inside = True
                        elif self._cSpace[i*self._bbSize+k]==True and inside==True:
                            hcross2 = k
                    #如果没有找到覆盖区域
                    if inside==False:continue
                    #填充覆盖区域
                    for j in range(self._bbSize):
                        if j>=hcross1 and j<=hcross2:
                            self._cSpace[i*self._bbSize+j]=True
                
                #设置表格
                for i in range(self._bbSize):
                    for j in range(self._bbSize):
                        if self._cSpace[i*self._bbSize+j]:
                            self._lookup[q*self._headings+o].append(Point2D(j-math.floor(self._c.x),\
                                                                                i-math.floor(self._c.y))) # 相较于中心的坐标
                # #debug
                # for i in range(self._bbSize):
                #     print(" ")
                #     for j in range(self._bbSize):
                #         if self._cSpace[i*self._bbSize+j]==True:
                #             print("#",end='')
                #         else:
                #             print("-",end='')


    """
    @brief collisionDetection--碰撞检测
    输入:x,y——与地图原点的相对坐标,h——偏航角,image——二值化地图(observation['hdmaps_info']['image_mask'].image_ndarray)
    """
    def collisionDetection(self,x,y,h,image):
        start_time = time.perf_counter()
   
        x = x/0.1   #self._cSize
        y = y/0.1   #self._cSize
        X = math.floor(x)
        Y = math.floor(y)
        iX = math.floor((x-X)*self._position_resolution)
        iY = math.floor((y-Y)*self._position_resolution)
        while h>math.pi*2:
            h-=math.pi*2
        while h<0:
            h+=math.pi*2
        iH = math.floor(h/self._delt_heading_rad)
        idx = iY*self._position_resolution*self._headings + iX*self._headings + iH

        for i in range(len(self._lookup[idx])):
            cX = X+self._lookup[idx][i].x*2
            cY = Y+self._lookup[idx][i].y*2
            # if cX>=0 and cX<image.shape[1] and cY>=0 and cY<image.shape[0]:
            #     print(image[image.shape[0]-22000][image.shape[1]-8500])
            if image[cY][cX]==False:
                return True
        
        # flag = False
        # for j in range(13220,13120,-1):
        #     for i in range(7476,7576):
            
        #         # for k in self._lookup[idx]:
        #         #     if i == X+k.x and j == Y+k.y:
        #         #         flag = True
        #         # if flag == True:
        #         #     if image[j][i] == True:
        #         #         print("1",end='')
        #         #     else:
        #         #         print('0',end='')
        #         # else:
        #         #     print(' ',end='')
        #         # flag = False
        #         if image[image.shape[0]-j][i] == True:
        #             print("1",end='')
        #         else:
        #             print('0',end='')
        #     print('---\n')
        end_time = time.perf_counter()
        # print("碰撞检测时间:"+str(end_time-start_time))
        return False

if __name__ == "__main__":
    start_time = time.perf_counter()
    lookup = CollisionLookup()
    end_time = time.perf_counter()
    print("运行时间:"+str(end_time-start_time))
        
