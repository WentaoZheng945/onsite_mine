import sys
import os
from pathlib import Path as PathlibPath
from typing import Dict,List,Tuple,Optional,Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# 第三方库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
import matplotlib.patches as patches
from matplotlib.patches import Rectangle,Arrow
from matplotlib.axes import Axes

# 自定义模块
from dynamic_scenes.observation import Observation
from map_expansion.map_api import TgScenesMap,TgScenesMapExplorer
from map_expansion.bit_map import BitMap



def check_dir(target_dir:str) -> None:
    """
    Check and create the directory if it doesn't exist.

    Parameters:
    - target_dir (str):The directory path to be checked.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)  # If directory already exists,don't raise an error.





class Visualizer:
    COLOR_MAP = {
        'background_vehicle':'#6495ED', # "cornflowerblue" 矢车菊蓝;矢车菊的蓝色
        'static_obstacle':'#4682B4',  # 静态障碍物，钢青色
        'ego_vehicle':'#008000',  # 深绿色，主车
        'warning_signal':'#e60000',  # 红色，警告标志
        'traj_prediction_line':'#4de680',  # 预测轨迹，绿松石色
        'traj_prediction_point':'#127436',  # 预测轨迹点，深绿色
        'traj_true_line':'#e60000',  # 红色，真实轨迹
        'traj_true_point':'#9B0000',  # 深红色，真实轨迹点
        'traj_true_history':'#eb984e',  # 历史轨迹，橙棕色
        'target_box_facecolor':'#FF6347'}  # 目标区域，橙红色
    X_MARGIN = 8 # 类常量
    Y_MARGIN = 8

    def __init__(self):
        self.flag_visilize = True # 是\否进行可视化绘图
        self.flag_hdmaps_visilized = False # hdmaps已经被绘图(仅绘制一次即可)标志位
        self.flag_visilize_ref_path = True #是否可视化参考路径
        self.flag_visilize_prediction = False #是否可视化轨迹预测结果
        self.flag_save_fig_whitout_show = False # 绘图保存,但是不显示
        

    def init(self,observation:"Observation",flag_visilize:bool = False,flag_save_fig_whitout_show:bool = False, img_save_path='') -> None:
        """绘图,第一次进入,初始化.

        Args:
            observation (Observation): 当前时刻对动态场景、静态地图的观测结果.
            flag_visilize (bool, optional): 是\否进行可视化绘图. Defaults to False.
        """
        # 测试信息
        self.scenario_name = observation.test_setting['scenario_name']
        self.scenario_type = observation.test_setting['scenario_type']
        self.save_path_img = os.path.join(img_save_path , f"image_{self.scenario_name}")
        
        # 可视化设置
        self.flag_visilize = flag_visilize
        self.flag_save_fig_whitout_show = flag_save_fig_whitout_show
        
        # 初始化画布
        if self.flag_visilize:
            if self.flag_save_fig_whitout_show == True:
                plt.ioff()  # 关闭交互模式，没有plt.show不显示（即不显示可视化）
            else:
                plt.ion()  # 交互模式，随时显示，不需要plt.show
            self.fig = plt.figure(figsize=(10.0,10.0))
            # self.fig = plt.figure(figsize=[5.0,5.0])
            self.axbg = self.fig.add_subplot()  # 多个x轴,共用y轴
            self.axveh = self.axbg.twiny()
            self.axTraj = self.axbg.twiny()
        else:
            plt.ioff()
            return

        self.flag_hdmaps_visilized = False # 本次绘制地图
        if observation.hdmaps:
            self.x_target = list(observation.test_setting['goal']['x'])
            self.y_target = list(observation.test_setting['goal']['y'])
            self.heading = observation.test_setting['goal']['head']
            # 获取绘图范围
            bitmap_info = observation.hdmaps['image_mask'].bitmap_local_info
            self.x_min = bitmap_info['utm_local_range'][0]
            self.x_max = bitmap_info['utm_local_range'][2]
            self.y_min = bitmap_info['utm_local_range'][1]
            self.y_max = bitmap_info['utm_local_range'][3]
            self.axbg.set_xlim(self.x_min - self.X_MARGIN,self.x_max + self.X_MARGIN)
        
        # 更新观测值 and 绘图刷新(第一次进入)
        self.update(observation,traj_future=100,observation_last=100,traj=100)


    def update(self,observation:Observation,traj_future,observation_last,traj) -> None:
        """更新观测值 and 绘图刷新.

        Args:
            observation (Observation): 当前时刻对动态场景、静态地图的观测结果.
            traj_future (_type_): 背景车预测的轨迹.
            observation_last (_type_): 上一刻 动态场景、静态地图的观测结果.
            traj (_type_):该场景背景车辆全部轨迹(ground truth).
        """
        # --------结束条件如果不画图, 直接退出 --------
        if not self.flag_visilize:
            return
        self.axveh.cla()  # 清除当前子图中的所有绘图内容
        self.axTraj.cla()  # 清除当前子图中的所有绘图内容;

        # -------- 绘制道路 \ 绘制车辆 \  --------
        if not self.flag_hdmaps_visilized:
            self._plot_hdmaps(observation)  # 绘制位图、语义地图的road、intersection、loading_area、unloading_area、dubins_pose以及reference_path
            self._plot_target_area(observation)  # 绘制目标区域(loading场景加载heading指向)
            self.flag_hdmaps_visilized = True  # 防止绘制多次
        self._plot_vehicles(observation)
        
        # --------若发生碰撞,则绘制碰撞警告标志--------
        if observation.test_setting['end'] in [2,3]:
            self._plot_warning_signal(observation)
        
        # --------绘制车辆轨迹:预测轨迹与真实轨迹--------
        if (traj_future != 100 and  traj != 100):#具备可视化 预测轨迹\真实轨迹 的条件
            if self.flag_visilize_prediction == True:
                self._plot_vehicles_traj(observation,traj_future,observation_last,traj)

        # --------画布属性设置--------
        # plt.rcParams['font.family'] = 'Times New Roman'  # 若需显示汉字 SimHei 黑体, STsong 华文宋体还有 font.style  font.size等
        plt.rcParams['axes.unicode_minus'] = False
        if observation.hdmaps:# 有地图文件,采用主车中心视角
            plt.ylim(self.y_min - self.Y_MARGIN,self.y_max + self.Y_MARGIN)   # 指定y坐标的显示范围,防止车辆畸变,y轴范围与x轴一致
            self.axbg.set_xlim(self.x_min - self.X_MARGIN,self.x_max + self.X_MARGIN)  
            self.axveh.set_xlim(self.x_min - self.X_MARGIN,self.x_max + self.X_MARGIN)
            self.axTraj.set_xlim(self.x_min - self.X_MARGIN,self.x_max + self.X_MARGIN)  
            
        else: # 无地图文件的(如加速测试),采用主车中心视角
            plt.gca().set_aspect('equal')
            x_center = observation.vehicle_info['ego']['x']
            plt.xlim(x_center - 20,x_center + 200)
            plt.ylim(-80,80)

        # -------- GIF显示各类信息 --------
        # 显示测试相关信息
        ts_text = "Time_stamp:" + str(round(observation.test_setting['t'],4))
        name_text = 'Test_scenario:' + str(self.scenario_name)
        type_text = 'Task_type:' + str(self.scenario_type)
        self.axveh.text(0.02,1.05,name_text,transform=self.axveh.transAxes,fontdict={'size':'10','color':'black'})  # axveh.transAxes表示坐标比例系
        self.axveh.text(0.02,1.08,type_text,transform=self.axveh.transAxes,fontdict={'size':'10','color':'black'})
        self.axveh.text(0.02,1.11,ts_text,transform=self.axveh.transAxes,fontdict={'size':'10','color':'black'})
        
        #  碰撞信息显示 
        if observation.test_setting['end'] in [2,3]:
            collision_text = 'A collision has occurred.'
            self.axveh.text(0.02,0.85,collision_text,transform=self.axveh.transAxes,fontdict={'size':'18','color':'red'})
        
        #  显示所有车辆运行信息 
        colLabels = list(observation.vehicle_info.keys())
        rowLabels = ['v (m/s)','a (m/s2)'] 
        v = np.array([round(observation.vehicle_info[key]['v_mps'],4) for key in colLabels]).reshape(1,-1)
        a = np.array([round(observation.vehicle_info[key]['acc_mpss'],4) for key in colLabels]).reshape(1,-1)
        cellTexts = np.vstack((v,a))
        info_table = self.axveh.table(cellText=cellTexts,colLabels=colLabels,rowLabels=rowLabels,
                                      rowLoc='center',colLoc='center',cellLoc='center',loc='bottom')
        info_table.auto_set_font_size(False)
        info_table.set_fontsize(10)

        # --------刷新当前帧画布--------
        plt.subplots_adjust()
        step_sum = int(round(observation.test_setting['max_t'] / observation.test_setting['dt']))
        if self.flag_save_fig_whitout_show:
            # 想要把预测结果保存为gif,先把png保存下来.
            step = int(round(observation.test_setting['t'] / observation.test_setting['dt'])) 
            png_dir = os.path.join(self.save_path_img,'images')
            check_dir(png_dir)
            plt.savefig(os.path.join(png_dir,f"image_{observation.test_setting['scenario_name']}_{step}.png"))
            # print(f"###保存单帧图片### 第{step}帧/最大帧数{step_sum}")
        else:
            plt.pause(1e-7)
            plt.show()
        
        # -------- 如果测试结束,则结束绘图,关闭绘图模块--------
        if observation.test_setting['end'] != -1:
            plt.ioff()
            plt.close()
            return


    def _plot_vehicles_traj(self,observation:Observation,traj_future:Dict,observation_last:Observation,traj:Dict) -> None:
        """绘制车辆预测轨迹及真实轨迹.

        Args:
            observation (Observation): 当前时刻对动态场景、静态地图的观测结果.
            traj_future (_type_): 背景车预测的轨迹.
            observation_last (_type_): 上一刻 动态场景、静态地图的观测结果.
            traj (_type_):该场景背景车辆全部轨迹(ground truth).
        """
         
        for id_vehi,traj_predict in traj_future.items():
            # Extract true trajectory
            traj_true = traj.get(id_vehi,{}).copy()
            
            # Remove unwanted keys from true trajectory
            traj_true.pop('shape',None)
            traj_true.pop(-1,None)  # 没用但不会报错
            
            # Ensure the vehicle ID exists in the true trajectory
            if not traj_true:
                raise ValueError(f"Error:No vehicle with ID '{id_vehi}' found in the true trajectory.")
            
            # Plot individual vehicle trajectory
            self._plot_single_vehicle_traj(observation,id_vehi,traj_true,traj_predict)

  
    def _plot_single_vehicle_traj(self,observation:Observation,id_vehi:int,traj_true:Dict,traj_predict:Dict)-> None:
        """ 利用 matplotlib 绘制单个车的轨迹.
        """
        # 1) 从字典中提取x y轨迹点的列表
        # 1.1) 计算key值列表
        # 获取字典的键并转换为浮点数
        keys_as_floats = [float(key) for key in traj_true.keys()] 
        max_time_of_id_vehi = max(keys_as_floats)

        dt = round(float(observation.test_setting['dt']),2) # 时间步长
        now_time = round(float(observation.test_setting['t'])-dt,2)   # 当前时间
        end_time = float(observation.test_setting['max_t'])                
        delta_t_1 = end_time - now_time
        delta_t_2 = max_time_of_id_vehi - now_time
        delta_t = min(delta_t_1,delta_t_2)
        
        if delta_t < 5.0 :
            numOfTrajPoint_ = int(delta_t / dt)
            float_list,keys_to_extract_line= self._generate_float_list(now_time,dt,numOfTrajPoint_)  # 返回时间戳列表
            float_list_1s,keys_to_extract_line_1s= self._generate_float_list_2(now_time,1.0,end_time)
        else:
            numOfTrajPoint_ = int(5.0/ dt) # 5秒*10hz;
            float_list,keys_to_extract_line= self._generate_float_list(now_time,dt,numOfTrajPoint_)
            float_list_1s,keys_to_extract_line_1s = self._generate_float_list_2(now_time,1.0,now_time+5.0)       

        # 1.2) 提取 5s的所有轨迹点 + 1 2 3 4 5秒处的轨迹点
        xlinelist_true,ylinelist_true =  self._find_xylist_from_vehi_traj(keys_to_extract_line,traj_true)
        xpointlist_true,ypointlist_true = self._find_xylist_from_vehi_traj(keys_to_extract_line_1s,traj_true)
        self._plot_traj_line_points(xlinelist=xlinelist_true,
                                    ylinelist=ylinelist_true,
                                    xpointlist=xpointlist_true,
                                    ypointlist=ypointlist_true,
                                    color_line=self.COLOR_MAP['traj_true_line'],
                                    line_wide=2.0,
                                    color_point=self.COLOR_MAP['traj_true_point'],
                                    size_point=4.0)
        # 2.2) plot 预测的轨迹
        xlinelist_predict,ylinelist_predict =  self._find_xylist_from_vehi_traj(keys_to_extract_line,traj_predict)
        xpointlist_predict,ypointlist_predict = self._find_xylist_from_vehi_traj(keys_to_extract_line_1s,traj_predict)
        self._plot_traj_line_points(xlinelist=xlinelist_predict,
                                    ylinelist=ylinelist_predict,
                                    xpointlist=xpointlist_predict,
                                    ypointlist=ypointlist_predict,
                                    color_line=self.COLOR_MAP['traj_prediction_line'],
                                    line_wide=1.5,
                                    color_point=self.COLOR_MAP['traj_prediction_point'],
                                    size_point=3.0)
    
    
    def _plot_traj_line_points(self,xlinelist,ylinelist,xpointlist,ypointlist,color_line,line_wide,color_point,size_point):
        """绘制轨迹线 and 轨迹点.
        """
        self.axTraj.plot(
            xlinelist,
            ylinelist,
            color=color_line,
            lw=line_wide )
        self.axTraj.scatter(
            xpointlist,
            ypointlist,
            color=color_point,
            s=size_point )
         
            
    def _generate_float_list(self,start,step,count):
        """生成时间序列.
        """
        float_list = []
        float_list_tostr = []
        current_value = start
        for _ in range(count):
            float_list.append(round(current_value,2))
            float_list_tostr.append(str(round(current_value,2)))
            current_value += step
        return float_list,float_list_tostr
    
    
    def _generate_float_list_2(self,start,step,end):
        """生成时间序列,case 2:1s短时预测.
        """
        float_list = []
        float_list_tostr = []
        current_value = start
        while current_value <= end:
            float_list.append(round(current_value,2))
            float_list_tostr.append(str(round(current_value,2)))
            current_value += step
        return float_list,float_list_tostr
    
    
    def _find_xylist_from_vehi_traj(self,strkey_list,vehi_traj):
        """提取 5s的所有轨迹点 and 1 2 3 4 5秒处的轨迹点.
        """
        value_list_x = []
        value_list_y = []  
        for key in strkey_list:
            if key in vehi_traj:
                value_list_x.append(  vehi_traj[key]['x']  )
                value_list_y.append(  vehi_traj[key]['y']  )
        return value_list_x,value_list_y    


    def _plot_vehicles(self,observation:Observation) -> None:
        """绘制all车辆BOX.
        """
        for key,values in observation.vehicle_info.items():
            if key == 'ego':
                self._plot_single_vehicle(key,values,c=self.COLOR_MAP['ego_vehicle'])
            else:
                if values['shape']['vehicle_type'] == 'rock':
                    self._plot_single_rock(key, values, c=self.COLOR_MAP['static_obstacle'])  # 静态障碍物
                else:
                    self._plot_single_vehicle(key,values,c=self.COLOR_MAP['background_vehicle'])

    def _plot_single_rock(self, key: str, vehi: dict, c=None):
        """绘制单个车辆BOX.
        注:利用 matplotlib 和 patches 绘制小汽车,以 x 轴为行驶方向
        """
        x = vehi['x']
        y = vehi['y']
        yaw = vehi['yaw_rad']

        x_A3 = x - vehi['shape']['locationPoint2Rear'] * np.cos(yaw) + 0.5 * vehi['shape']['width'] * np.sin(yaw)
        y_A3 = y - vehi['shape']['locationPoint2Rear'] * np.sin(yaw) - 0.5 * vehi['shape']['width'] * np.cos(yaw)
        width_x = vehi['shape']['length']  # 绘图,x轴方向定义为宽
        height_y = vehi['shape']['width']  # 绘图,y轴方向定义为高

        self.axveh.add_patch(
            patches.Rectangle(
                xy=(x_A3, y_A3),  # 矩形的左下角坐标
                width=width_x,
                height=height_y,
                angle=yaw / np.pi * 180,
                color=c,
                fill=True,
                zorder=3))
        if key != 'ego':
            self.axveh.annotate('r', (x, y))

    def _plot_single_vehicle(self,key:str,vehi:dict,c=None):
        """绘制单个车辆BOX.
        注:利用 matplotlib 和 patches 绘制小汽车,以 x 轴为行驶方向
        """
        x = vehi['x']
        y = vehi['y']
        yaw = vehi['yaw_rad']
        
        x_A3 = x - vehi['shape']['locationPoint2Rear'] * np.cos(yaw) + 0.5* vehi['shape']['width'] *np.sin(yaw)
        y_A3 = y - vehi['shape']['locationPoint2Rear'] * np.sin(yaw) - 0.5* vehi['shape']['width'] *np.cos(yaw)
        width_x= vehi['shape']['length']  # 绘图,x轴方向定义为宽
        height_y= vehi['shape']['width']   # 绘图,y轴方向定义为高

        self.axveh.add_patch(
            patches.Rectangle(
                xy=(x_A3,y_A3),#矩形的左下角坐标
                width=width_x,
                height=height_y,
                angle=yaw / np.pi * 180,
                color=c,
                fill=True,
                zorder=3  ))
        if key != 'ego':
            self.axveh.annotate(key,(x,y))


    def _plot_warning_signal(self,observation:Observation ):
        """绘制主车碰撞时的提醒标志.
        """
        for key,values in observation.vehicle_info.items():
            if key == 'ego':
                x,y = [float(values[i]) for i in ['x','y']]
                self.axveh.scatter(x,y,s=60,c=self.COLOR_MAP['warning_signal'],alpha=1.0,marker=(8,1,30),zorder=4)


    def _plot_target_area(self,observation:Observation):
        """绘制该场景预设目标区域.
        """        
        if self.x_target and self.y_target:
            x,y = self.x_target,self.y_target
            codes_box = [MatplotlibPath.MOVETO] + [MatplotlibPath.LINETO] * 3 + [MatplotlibPath.CLOSEPOLY] 
            vertices_box = [(self.x_target[0],self.y_target[0]),
                            (self.x_target[1],self.y_target[1]),
                            (self.x_target[2],self.y_target[2]),
                            (self.x_target[3],self.y_target[3]),
                            (self.x_target[0],self.y_target[0])] # 4 points of polygon
            path_box = MatplotlibPath(vertices_box,codes_box)  # 定义对应Path
            pathpatch_box = patches.PathPatch(path_box,
                                              facecolor=self.COLOR_MAP['target_box_facecolor'],
                                              edgecolor='orangered',
                                              zorder=2,
                                              alpha=0.7)
            self.axbg.add_patch(pathpatch_box)
            if self.scenario_type == 'loading' and self.heading is not None:
                center_x,center_y = sum(x)/len(x),sum(y)/len(y)
                node_ux = 5*np.cos(self.heading[0])
                node_vy = 5*np.sin(self.heading[0])
                self.axbg.arrow(center_x, center_y, node_ux, node_vy, color='red', alpha=1, head_width=2.0, head_length=2.0)
            
            
    def _plot_hdmaps(self,observation:Observation,) -> None:
        """根据observation绘制地图.
        注:只要完成绘制工作即可.plt.plot().其他plt.show()之类的不需要添加.

        Args:
            observation (Observation): 当前时刻的观察值
        """
        if not observation.hdmaps:
            return
        # plot mask图,plot 道路片段、交叉口多边形划分;
        my_patch = (self.x_min,self.y_min ,self.x_max,self.y_max)  #  (x_min,y_min,x_max,y_max).
        # layer_names = ['intersection','road']
        layer_names =  ['road','intersection','loading_area','unloading_area','road_block']
        self._plot_hdmaps_render_map_patch(tgsc_map_explorer=observation.hdmaps['tgsc_map'].explorer,  # TgScenesMapExplorer类
                                           box_coords=my_patch,layer_names=layer_names,alpha=0.7,  # 坐标范围，需要绘制的层，渗透率
                                           render_egoposes_range=False,render_legend=False,
                                           bitmap=observation.hdmaps['image_mask'],  # Bitmap类
                                           # bitmap=observation.hdmaps['image_rgb'],
                                           ax= self.axbg)
        
        
    def _plot_hdmaps_render_map_patch(self,
                                    tgsc_map_explorer:TgScenesMapExplorer =None,
                                    box_coords:Tuple[float,float,float,float]=(0,0,1,1),
                                    layer_names:List[str] = None,
                                    alpha:float = 0.5,
                                    render_egoposes_range:bool = False,
                                    render_legend:bool = False,
                                    bitmap:Optional[BitMap] = None,
                                    ax:Axes = None) -> None:                        
        """ 渲染一个矩形图框,指定矩形框的xy坐标范围.By default renders all layers.        
        """
        x_min,y_min,x_max,y_max = box_coords
        local_width = x_max - x_min
        local_height = y_max - y_min

        # 渲染位图
        if bitmap is not None:
            if bitmap.bitmap_type == 'bitmap_mask':
                bitmap.render_mask_map_using_image_ndarray_local(ax,window_size=5,gray_flag=False) #!mask图降采样绘图
            elif bitmap.bitmap_type == 'bitmap_rgb':
                bitmap.render_rgb_map(ax)
            else:
                raise Exception('###Exception### 非法的 bitmap type:%s' % self.bitmap_type) # 自定义异常
            
        for layer_name in layer_names:  # 渲染各图层（5个）
            tgsc_map_explorer._render_layer(ax,layer_name,alpha)
         
        #  渲染intersection的reference path,lane（交叉口中）
        if self.flag_visilize_ref_path == True:
            tgsc_map_explorer.render_connector_path_centerlines(ax,alpha,resampling_rate=0.05)  # 20个点去一个绘图
        
        #  渲染road的reference path,lane（道路内）
        if self.flag_visilize_ref_path == True:
            tgsc_map_explorer.render_base_path_centerlines(ax,alpha,resampling_rate=0.2)  # 5个点取一个绘图
        
        if render_egoposes_range:
            ax.add_patch(Rectangle((x_min,y_min),local_width,local_height,fill=False,linestyle='-.',color='red',lw=3))
            ax.text(x_min + local_width / 100,y_min + local_height / 2,"%g m" % local_height,
                    color='red',fontsize=14,weight='bold')
            ax.text(x_min + local_width / 2,y_min + local_height / 100,"%g m" % local_width,
                    color='red',fontsize=14,weight='bold') #fig上添加文字标注
        if render_legend:
            ax.legend(frameon=True,loc='upper right')
  
  
  
  
    
