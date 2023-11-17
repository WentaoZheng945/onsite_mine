# Code written by Chen Zhifa(陈志发,chenzhifa@buaa.edu.cn),2023.
# 内置库 
import sys
import os
import json
import math
from typing import Dict,List,Tuple,Optional,Union

# 第三方库
import numpy as np
import descartes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from shapely import affinity
from shapely.geometry import Polygon,LineString,Point,box

# # 添加必要的路径到sys.path
def add_path_to_sys(target:str):
    abs_path = os.path.abspath(target)
    if abs_path not in sys.path:
        sys.path.append(abs_path)
dir_current_file = os.path.dirname(__file__)  # devkit/onsite_mine_code/map_expansion
dir_parent_1 = os.path.dirname(dir_current_file) # devkit/onsite_mine_code
dir_parent_2 = os.path.dirname(dir_parent_1) # devkit 
add_path_to_sys(dir_current_file)
add_path_to_sys(dir_parent_1)


# 自定义库
from map_expansion.bit_map import BitMap
import common.utils as utils

# Define a map geometry type for polygons and lines.
Geometry = Union[Polygon,LineString]
locations = ['jiangxi_jiangtong','guangdong_dapai']



class TgScenesMap:
    """用于从语义地图中查询和检索多个图层信息.
    每张局部地图 有精准的原点(西南角,[lat,lon]):
    我们使用与谷歌Maps/Earth相同的WGS 84 Web Mercator (EPSG:3857)投影."CRS_name":"GCS_WGS_1984_UTM_Zone_50N",
    """
    def __init__(self,
                 dataroot:str='/home/czf/project_czf/20231010_onsite_mine/datasets',
                 location:str = 'jiangxi_jiangtong'):
        """加载 HD MAP 的多个图层数据,并初始化explorer class.

        Args:该函数接受两个参数
            dataroot (str, optional): datasets根目录. Defaults to '/home/czf/project_czf/20231010_onsite_mine/datasets'.
            location (str, optional): 具体矿区位置的名称,当前有以下可选 :`jiangxi_jiangtong`,`guangdong_dapai`. Defaults to 'jiangxi_jiangtong'.

        Raises:
            Exception: xxxx_semantic_map.json version is error.
        """
       
        assert location in locations,'Error:地图名字错误:%s!' % location

        self.dataroot = dataroot
        self.location = location
        
        self.geometric_layers = ['polygon','node','node_block']
        self.other_layers = ['dubins_pose','road_block'] 
        self.non_geometric_line_layers = ['reference_path','borderline']
        self.non_geometric_polygon_layers = ['road','intersection','loading_area','unloading_area']
        
        self.layer_names = self.geometric_layers + self.non_geometric_polygon_layers +self.other_layers 

        # 加载高精度矢量语义地图
        semantic_map_hashes = {
                "jiangxi_jiangtong":'jiangxi_jiangtong_semantic_map',
                "guangdong_dapai":'guangdong_dapai_semantic_map'  }
        semantic_map_hash = semantic_map_hashes[self.location]
        dir_semantic_map =  os.path.join(self.dataroot, "semantic_map" ,semantic_map_hash+'.json')
        with open(dir_semantic_map,'r') as f:
            self.semantic_map_json = json.load(f)

        # check the map version.
        if 'version' in self.semantic_map_json:
            self.version = self.semantic_map_json['version']
        else:
            self.version = '1.5'
        if self.version != '1.5':
            raise Exception('Error:只针对地图版本为 map version 1.5! 你的地图版本是',self.version)
        
        # load json data
        self._load_layers() # 加载各个图层 
        self._make_token2ind() # token标识符转化为id: 0,1,2, ...... ;
        

        self.explorer = TgScenesMapExplorer(self)  # 确定横纵比


    def _load_layer(self,layer_name:str) -> List[dict]:
        """Returns a list of info corresponding to the layer name. 返回各个图层 层名 对应的列表.

        Args:
            layer_name (str): Name of the layer that will be loaded.

        Returns:
            List[dict]: A list of info corresponding to a layer.
        """
        
        return self.semantic_map_json[layer_name]


    def _load_layers(self) -> None:
        """ Loads each available layer. 
        """
        self.node = self._load_layer('node')
        self.node_block = self._load_layer('node_block')
        self.polygon = self._load_layer('polygon')
        self.road= self._load_layer('road')
        self.intersection= self._load_layer('intersection')
        self.loading_area= self._load_layer('loading_area')
        self.unloading_area= self._load_layer('unloading_area')
        self.road_block = self._load_layer('road_block')
        self.dubins_pose = self._load_layer('dubins_pose')
        self.reference_path = self._load_layer('reference_path')
        self.borderline = self._load_layer('borderline')   
        

    def _make_token2ind(self) -> None:
        """ Store the mapping from token to layer index for each layer.
        token2ind = {
        "node": {"node-0": 0,
		         "node-1": 1
		        },
        "node_block": {"nodeblock-0": 0,
			           "nodeblock-1": 1,
			          },
        }
        """
        self.token2ind = dict()
        for layer_name in self.layer_names:
            if layer_name == 'loading_area':  # only one
                continue
            self.token2ind[layer_name] = dict()
            for index,member in enumerate(getattr(self, layer_name)):
                self.token2ind[layer_name][member['token']] = index  # as: token2ind['node']['node_0'] = 0


    def get(self, layer_name: str, token: str) -> dict:
        """
        Returns a record from the layer in constant runtime.在固定的运行时间内从层中返回一条记录.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: A single layer record.
        """
        assert layer_name in self.layer_names, "Layer {} not found".format(layer_name)

        return getattr(self, layer_name)[self.getind(layer_name, token)]
    
    
    def getind(self, layer_name: str, token: str) -> int:
        """
        This returns the index of the record in a layer in constant runtime. .在固定的运行时间内从层中返回一条记录的id.
        :param layer_name: Name of the layer we are interested in.
        :param token: Token of the record.
        :return: The index of the record in the layer, layer is an array.
        """
        return self.token2ind[layer_name][token]


    def render_layers(self,
                      layer_names:List[str],
                      alpha:float = 0.5,
                      figsize:Union[None,float,Tuple[float,float]] = None,
                      tokens:List[str] = None,
                      bitmap:Optional[BitMap] = None) -> Tuple[Figure,Axes]:
        """
        Render a list of layer names.
        :param layer_names:A list of layer names.
        :param alpha:The opacity of each layer that gets rendered.
        :param figsize:Size of the whole figure.
        :param tokens:Optional list of tokens to render. None means all tokens are rendered.
        :param bitmap:Optional BitMap object to render below the other map layers.
        :return:The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_layers(layer_names,alpha,
                                           figsize=figsize,tokens=tokens,bitmap=bitmap)

    def render_map_patch(self,
                         box_coords:Tuple[float,float,float,float],
                         layer_names:List[str] = None,
                         alpha:float = 0.5,
                         figsize:Tuple[int,int] = (15,15),
                         render_egoposes_range:bool = True,
                         render_legend:bool = True,
                         bitmap:Optional[BitMap] = None,
                         fig:plt.figure = None) -> Tuple[Figure,Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords:The rectangular patch coordinates (x_min,y_min,x_max,y_max).
        :param layer_names:All the non geometric layers that we want to render.
        :param alpha:The opacity of each layer.
        :param figsize:Size of the whole figure.
        :param render_egoposes_range:Whether to render a rectangle around all ego poses.
        :param render_legend:Whether to render the legend of map layers.
        :param bitmap:Optional BitMap object to render below the other map layers.
        :return:The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_map_patch(box_coords,layer_names=layer_names,alpha=alpha,figsize=figsize,
                                              render_egoposes_range=render_egoposes_range,
                                              render_legend=render_legend,bitmap=bitmap,fig = fig)

    def render_bitmap_mask(self,
                            patch_box_meter:Tuple[float,float,float,float],
                            patch_angle:float,
                            layer_names:List[str] = None,
                            figsize:Tuple[int,int] = (15,15),
                            n_row:int = 2) -> Tuple[Figure,List[Axes]]:
        """
        Render mask黑白掩码图像.
        :param Patch box :定义显示图框(补丁框)大小 [x_center,y_center,height,width],UTM坐标系,单位:米.
        :param patch_angle: 显示图框(补丁框)的方向角,单位:deg,正北为0 deg.
        :param layer_names:['intersection','road']或者 None
        :param canvas_size_pixel:Size of the outputs mask (h,w).
        :param figsize:Size of the figure.
        :param n_row:Number of rows with plots. 绘图的行数
        :return:The matplotlib figure and a list of axes of the rendered layers.
        """
        # 计算 patch_box_meter对应像素的宽*高,单位pix
        canvas_size_pixel = np.array((patch_box_meter[2],patch_box_meter[3])) * self.scale_PixelPerMeter
        canvas_size_pixel = tuple(np.round(canvas_size_pixel).astype(np.int32))


        return self.explorer.render_bitmap_mask(patch_box_meter,patch_angle,
                                                layer_names=layer_names,canvas_size_pixel=canvas_size_pixel,
                                                figsize=figsize,n_row=n_row)

    def render_bitmap_rgb(self,
                            patch_box_meter:Tuple[float,float,float,float],
                            patch_angle:float,
                            layer_names:List[str] = None,
                            figsize:Tuple[int,int] = (15,15),
                            n_row:int = 2) -> Tuple[Figure,List[Axes]]:
        """
        Render RGB-PNG图像 of the patch specified by patch_box_meter and patch_angle.
        :param Patch patch_box_meter_meter :定义显示图框(补丁框)大小 [x_center,y_center,height,width],UTM坐标系,单位:米.
        :param patch_angle: 显示图框(补丁框)的方向角,单位:deg,正北为0 deg.
        :param layer_names:['intersection','road']或者 None
        :param canvas_size_pixel:Size of the outputs mask (h,w).
        :param figsize:Size of the figure.
        :param n_row:Number of rows with plots. 绘图的行数
        :return:The matplotlib figure and a list of axes of the rendered layers.
        """
        # 计算 patch_box_meter对应像素的宽*高,单位pix
        canvas_size_pixel = np.array((patch_box_meter[2],patch_box_meter[3])) * self.scale_PixelPerMeter
        canvas_size_pixel = tuple(np.round(canvas_size_pixel).astype(np.int32))


        return self.explorer.render_bitmap_rgb(patch_box_meter,patch_angle,
                                                layer_names=layer_names,canvas_size_pixel=canvas_size_pixel,
                                                figsize=figsize,n_row=n_row)

    def extract_polygon(self,polygon_token:str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token.用多边形标记构造一个形状良好的多边形对象.
        :param polygon_token:The token of the polygon record.
        :return:The polygon wrapped in a shapely Polygon object.
        """
        return self.explorer.extract_polygon(polygon_token)  # 得到对应的多边形

    
    def get_polygon_token_using_node(self,x:float,y:float,polygons:Dict=None) -> str:
        """依据(x,y)定位该点属于哪个多边形内部

        Args:
            x (float):_description_
            y (float):_description_
            polygons (Dict,optional):_description_. Defaults to None.

        Returns:
            _type_:多边形 token
        """
        # 逐个检查多边形
        if polygons is None:
            polygons =self.polygon
        for polygon in polygons:
            link_node_tokens = polygon["link_node_tokens"]
            # 将多边形的节点坐标提取为NumPy数组
            polygon_nodes = np.array([(node["x"],node["y"]) for node_token in link_node_tokens
                                    for node in self.node if node["token"] == node_token])
            # 使用了射线法来确定点是否在多边形内部.该实现假设多边形的边界是封闭的,即首尾节点连接形成闭合的多边形 
            if self._is_point_in_polygon(x,y,polygon_nodes):
                return polygon["token"]
            
        # 如果点不在任何一个多边形内部,引发一个异常
        raise Exception(f"({x},{y})不属于任何 polygon,请检查.")
    
    
    def get_dubinspose_token_from_polygon(self,veh_pose:Tuple[float,float,float],polygon_token:str) -> str:
        """根据veh_pose(x,y,yaw)车辆定位位姿,从polygon_token所属的dubinspose list中匹配lane_connector的起始dubinspose.

        前提:
        1. vehicle所处的polygon为进入交叉路口前的一个road多边形.
        2. veh_pose在polygon内部.

        匹配原则:
        - Step 1:筛除掉|航向夹角|>0.5*Pi.
        - Step 2:
        - Case1:对于剩下2个dubinspose情况:
            - Case1-1:longitudinal_error 全部>=0,选择|longitudinal_error|大的.
            - Case1-2:longitudinal_error一正一负,选择longitudinal_error正的.
            - Case1-3:longitudinal_error 全部<=0,选择|longitudinal_error|小的.
        - Case2:对于剩下1个dubinspose情况——选择该dubinspose.

        Args:
            veh_pose (Tuple[float,float,float]):车辆的位姿.
            polygon_token (str):指定的polygon token.

        Returns:
            str:最佳匹配的dubinspose_token.
        """

        if not polygon_token.startswith('polygon-'):
            raise ValueError(f"Invalid polygon_token:{polygon_token}")

        id_polygon = int(polygon_token.split('-')[1])
        if id_polygon > len(self.polygon):
            raise IndexError(f"Polygon ID {id_polygon} out of bounds.请检查.")

        link_dubinspose_tokens = self.polygon[id_polygon]['link_dubinspose_tokens']
        if not link_dubinspose_tokens:
            raise ValueError('No dubinspose tokens available for the specified polygon.')

        dubinsposes_indicators = []
        for token in link_dubinspose_tokens:
            id_dubinspose = int(token.split('-')[1])
            lateral_error,longitudinal_error,heading_error_rad = utils.compute_two_pose_error(
                self.dubins_pose[id_dubinspose]['x'],
                self.dubins_pose[id_dubinspose]['y'],
                self.dubins_pose[id_dubinspose]['yaw'],
                *veh_pose) # 每个dubinspose到veh_pose的关系
            
            distance = math.sqrt(lateral_error**2 + longitudinal_error**2)
            indicators = {
                "token_dubinspose":token,
                "lateral_error":lateral_error,
                "longitudinal_error":longitudinal_error,
                "heading_error_rad":heading_error_rad,
                "distance":distance
            }# 四个位姿关系指标
            dubinsposes_indicators.append(indicators)

        filtered_dubinsposes = [i for i,indicator in enumerate(dubinsposes_indicators)
                            if abs(indicator['heading_error_rad']) <= 0.5 * math.pi]

        if len(filtered_dubinsposes) == 1:
            return dubinsposes_indicators[filtered_dubinsposes[0]]["token_dubinspose"]
        elif len(filtered_dubinsposes) == 2:
            longitudinal_errors = dict()
            for index in filtered_dubinsposes:
                        longitudinal_errors[index] =  dubinsposes_indicators[index]['longitudinal_error']
            # 根据三种情况选择最佳的dubinspose
            if all(value >= 0 for value in longitudinal_errors.values()):
                 # case1-1:longitudinal_error 全部>=0,选择 |longitudinal_error|大的
                best_index = max(longitudinal_errors,key=lambda k:abs(longitudinal_errors[k]))

            elif all(value <= 0 for value in longitudinal_errors.values()):
                # case1-3:longitudinal_error 全部<=0,选择 |longitudinal_error|小的key
                best_index = min(longitudinal_errors,key=lambda k:abs(longitudinal_errors[k]))
            else:
                # case1-2:longitudinal_error 一正一负,选择 longitudinal_error 正的
                best_index = next(index for index,value in longitudinal_errors.items() if value > 0)

            return dubinsposes_indicators[best_index]["token_dubinspose"]

        raise ValueError(f"Unexpected condition with filtered node poses. Please check.")
        
        
        
    def get_dubinspose_token_from_polygon_old(self,veh_pose:Tuple[float,float,float],polygon_token:str)-> str:
        """功能:依据veh_pose(x,y,yaw)车辆定位位姿,从polygon_token所属的dubinspose list中匹配lane_connector的起始 dubinspose
        前提:1)vehicle 所处的polygon为进入交叉路口前的一个road 多边形.
                2)veh_pose在polygon内部.
        原理:匹配原则 
                -step 1:筛除掉 |航向夹角|>0.5*Pi;
                -step 2:-case1:对于剩下2个dubinspose情况:
                            -case1-1:longitudinal_error 全部>=0,选择 |longitudinal_error|大的;(无全部为0情况)
                            -case1-2:longitudinal_error 一正一负,选择 longitudinal_error 正的;
                            -case1-3:longitudinal_error 全部=<0,选择 |longitudinal_error|大的;(无全部为0情况)
                        -case2:对于剩下1个dubinspose情况——选择该dubinspose
        """       
        if polygon_token.startswith('polygon-'):
            id_polygon = int(polygon_token.split('-')[1])
            if id_polygon <=len(self.polygon):
                link_dubinspose_tokens = self.polygon[id_polygon]['link_dubinspose_tokens']
                if len(link_dubinspose_tokens) == 0:
                    raise Exception('No dubinspose tokens available for the specified polygon.')
                
                dubinspose_token_best_matched = None
                dubinsposes_indicators = [{} for _ in range(len(link_dubinspose_tokens))]
                for index,value in enumerate(link_dubinspose_tokens):
                    id_dubinspose = int(value.split('-')[1])
                    lateral_error,longitudinal_error,heading_error_rad=utils.compute_two_pose_error(
                                                    self.dubins_pose[id_dubinspose]['x'],
                                                    self.dubins_pose[id_dubinspose]['y'],
                                                    self.dubins_pose[id_dubinspose]['yaw'],
                                                    veh_pose[0],veh_pose[1],veh_pose[2]) #每个dubinspose到veh_pose的关系
                    distance = math.sqrt(lateral_error ** 2 + longitudinal_error ** 2)
                    dubinsposes_indicators[index]= {"token_dubinspose":value,
                                                "lateral_error":lateral_error,
                                                "longitudinal_error":longitudinal_error,
                                                "heading_error_rad":heading_error_rad,
                                                "distance":distance 
                                                }# 四个位姿关系指标
                
                filtered_dubinsposes = [ i for i,indicator in enumerate(dubinsposes_indicators)
                                    if abs(indicator['heading_error_rad']) <= 0.5 * math.pi]
                
                if len(filtered_dubinsposes)==1:
                    dubinspose_token_best_matched = dubinsposes_indicators[filtered_dubinsposes[0]]["token_dubinspose"]    
                
                elif len(filtered_dubinsposes)==2:
                    # 获取所有的longitudinal_error
                    longitudinal_errors = dict()
                    # longitudinal_errors = [dubinsposes_indicators[index]['longitudinal_error'] for index in filtered_dubinsposes]
                    for index in filtered_dubinsposes:
                        longitudinal_errors[index] =  dubinsposes_indicators[index]['longitudinal_error']
                        
                    
                    # 判断经过筛选后的longitudinal_error是否都是正的或都是负的
                    all_positive = all(err >= 0 for err in longitudinal_errors.values())
                    all_negative = all(err <= 0 for err in longitudinal_errors.values())

                    # 根据三种情况选择最佳的dubinspose
                    if all_positive:
                        # case1-1:longitudinal_error 全部>=0,选择 |longitudinal_error|大的
                        best_index = max(longitudinal_errors,key=longitudinal_errors.get)
                    elif all_negative:
                        # case1-3:longitudinal_error 全部<=0,选择 |longitudinal_error|大的key
                        best_index = min(longitudinal_errors,key=longitudinal_errors.get)
                    else:
                        # case1-2:longitudinal_error 一正一负,选择 longitudinal_error 正的
                        best_index = [index for index,err in longitudinal_errors.items() if err > 0][0]
                    dubinspose_token_best_matched = dubinsposes_indicators[best_index]["token_dubinspose"]    
                    
                else:
                    raise Exception(f"dubinspose_token_best_matched ={dubinspose_token_best_matched}有问题,请检查.") 
                if dubinspose_token_best_matched == None:
                    raise Exception(f"dubinspose_token_best_matched ={dubinspose_token_best_matched}有问题,请检查.")
                return dubinspose_token_best_matched
            
            else:
                raise Exception(f"polygon_token ={polygon_token},id_polygon ={id_polygon}超出界限,请检查.")
        else:
            raise Exception(f"polygon_token ={polygon_token}有问题,请检查.")
        
    @staticmethod
    def _is_point_in_polygon(x,y,polygon_nodes):
        # 将点坐标组合为一个坐标数组
        point = np.array([x,y])
        # 创建一个线段列表,每个线段连接多边形的两个相邻节点
        segments = [(polygon_nodes[i],polygon_nodes[(i + 1) % len(polygon_nodes)]) for i in range(len(polygon_nodes))]
        # 射线法判断点是否在多边形内部
        inside = False
        for segment in segments:
            # 提取线段的两个端点
            p1,p2 = segment
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                # 计算线段与射线的交点的x坐标
                x_intersect = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]) + p1[0]
                if x < x_intersect:
                    inside = not inside
        return inside
    
    @staticmethod
    def _calculate_distance_of_two_nodes(node1:Tuple[float,float],node2:Tuple[float,float]) -> float:
        
        x1,y1 = node1
        x2,y2 = node2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
        
    
    def _get_start_dubinspose_of_all_intersection(self,):
        pass
    
    # TODO
    def get_outgoing_plogon_token(self,):
        pass
    
    # TODO
    def get_outgoing_dubinspose_token(self,):
        pass
    
    # TODO
    def get_outgoing_dubinspose_token(self,):
        pass
    
    # TODO 增加新的脚本,处理所有的road中lane的连接,并将所有的path修改为lane;1) lane connectors ,交叉路口连通性;2)lane;
    # TODO 注意,nuscenes中 有关于 离散化采样 lane centerlines的代码 nuscenes-devkit/python-sdk/nuscenes/map_expansion/arcline_path_utils.py
    
    # TODO 
    def render_centerlines_all(self,
                            resolution_meters:float = 0.5,
                            figsize:Union[None,float,Tuple[float,float]] = None,
                            bitmap:Optional[BitMap] = None) -> Tuple[Figure,Axes]:
        """
        Render the centerlines of all lanes and lane connectors.
        渲染 所有的 path 和 connectivity_pathsample 的中心线centerlines;
        TODO 后续再实现渲染部分token确定的 lane centerlines
        """
        # 渲染 all lanes
        # self.explorer.render_centerlines(resolution_meters=resolution_meters,figsize=figsize,bitmap=bitmap)
        # 渲染 all lane connectors
        self.explorer.render_connector_path_centerlines(resolution_meters=resolution_meters,figsize=figsize,bitmap=bitmap)
    
    def _extract_connector_path(self)-> Tuple[list,list]:
        """ 
        """
        temp_reference_path = list()
        temp_reference_path = self.connectivity_path.copy()
        temp_intersection = list()
        temp_intersection = self.intersection.copy()
        temp_dubins_pose = list()
        temp_dubins_pose = self.dubins_pose.copy()
                
        for pathsample_key,pathsample_value in self.connector_path.items():
            # 1)处理 temp_reference_path
            pathsample_number = int( pathsample_key.split('-')[1]) 
            path_token = 'path-' + str(pathsample_number)  # 构建对应的path的token
            if 'pathsample' not in temp_reference_path[pathsample_number].keys():
                # 将pathsample的value加入到connectivity_path中对应的pathsample_number
                temp_reference_path[pathsample_number]['pathsample'] =dict()
                temp_reference_path[pathsample_number]['pathsample']['link_insection_token']=pathsample_value['link_insection_token']
                temp_reference_path[pathsample_number]['pathsample']['dubins_curve']=pathsample_value['dubins_curve']
                temp_reference_path[pathsample_number]['pathsample']['path_points']=np.array(pathsample_value['path_points'])
            
            #  2) 处理 temp_intersection
            intersection_number = int( pathsample_value['link_insection_token'].split('-')[1])# 'intersection-0'提取0
            if 'link_connectivity_path_tokens' not in temp_intersection[intersection_number]:
                temp_intersection[intersection_number]['link_connectivity_path_tokens'] = list() 
            temp_intersection[intersection_number]['link_connectivity_path_tokens'].append(path_token) 
            
            # 3) 处理 temp_dubins_pose
            start_dubinspose_token = pathsample_value['dubins_curve'][0]['start_dubinspose_token']
            start_dubinspose_number = int( start_dubinspose_token.split('-')[1])
            if self.dubins_pose[start_dubinspose_number]['is_first_dubinsnode']  is not True:
                raise Exception(f"dubins_pose 图层,token= {start_dubinspose_token} 有问题,请检查.")
            if 'link_connectivity_path_tokens' not in temp_dubins_pose[start_dubinspose_number]:
                temp_dubins_pose[start_dubinspose_number]['link_connectivity_path_tokens'] = list() 
            temp_dubins_pose[start_dubinspose_number]['link_connectivity_path_tokens'].append(path_token)
            
        return temp_reference_path,temp_intersection,temp_dubins_pose
        
        
        
        

class TgScenesMapExplorer:
    """ Helper class to explore the tgScenes map data. """
    def __init__(self,
                 map_api:TgScenesMap,
                 representative_layers:Tuple[str] = ('intersection','road','unloading_area','loading_area'),
                 color_map:dict = None):
        """
        :param map_api:TgScenesMap database class.
        :param representative_layers:代表整个地图数据的图层.
        :param color_map:Color map.
        """
        # Mutable default argument.
        if color_map is None:
            color_map = dict(road='#b2df8a',
                             road_block='#2A2A2A',
                             intersection='#fb9a99',
                             loading_area='#6a3d9a',
                             unloading_area='#7e772e',
                             connector_path='#1f78b4',
                             base_path='#1f78b4',
                             dubins_pose='#064b7a')
        
        self.map_api = map_api
        self.representative_layers = representative_layers
        self.color_map = color_map

        self.canvas_max_x = self.map_api.semantic_map_json['bitmap_rgb_PNG']['canvas_edge_meter'][0]# 画布的地理尺寸
        self.canvas_min_x = 0
        self.canvas_max_y = self.map_api.semantic_map_json['bitmap_rgb_PNG']['canvas_edge_meter'][1]
        self.canvas_min_y = 0
        self.canvas_aspect_ratio = (self.canvas_max_x - self.canvas_min_x) / (self.canvas_max_y - self.canvas_min_y)#画布横纵比


    def render_bitmap_mask(self,
                        patch_box_meter:Tuple[float,float,float,float],
                        patch_angle:float,
                        layer_names:List[str],
                        canvas_size_pixel:Tuple[int,int],
                        figsize:Tuple[int,int],
                        n_row:int = 2) -> Tuple[Figure,List[Axes]]:
        """
        渲染栅格图中的 bitmap_rgb.png
        the patch specified by patch_box_meter and patch_angle.
        :param Patch box :定义显示图框(补丁框)大小 [x_center,y_center,height,width],单位:米.
        :param patch_angle: 显示图框(补丁框)的方向角,单位:deg,正北为0 deg.
        :param layer_names:A list of layer names to be extracted.
        :param canvas_size_pixel:Size of the outputs mask (width,height). 画布的像素尺寸,宽*高,单位 pix .
        :param figsize:Size of the figure.
        :param n_row:Number of rows with plots.
        :return:The matplotlib figure and a list of axes of the rendered layers.
        """
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        map_mask = self.get_bitmap_mask(patch_box_meter,patch_angle,layer_names,canvas_size)

        # If no canvas_size is specified,retrieve the default from the outputs of get_map_mask.
        if canvas_size is None:
            canvas_size = map_mask.shape[1:]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim(0,canvas_size[1])
        ax.set_ylim(0,canvas_size[0])

        n_col = len(map_mask) // n_row
        gs = gridspec.GridSpec(n_row,n_col)
        gs.update(wspace=0.025,hspace=0.05)
        for i in range(len(map_mask)):
            r = i // n_col
            c = i - r * n_col
            subax = plt.subplot(gs[r,c])
            subax.imshow(map_mask[i],origin='lower')
            subax.text(canvas_size[0] * 0.5,canvas_size[1] * 1.1,layer_names[i])
            subax.grid(False)

        return fig,fig.axes
    
     
    def render_layers(self,
                      layer_names:List[str],
                      alpha:float,
                      figsize:Union[None,float,Tuple[float,float]],
                      tokens:List[str] = None,
                      bitmap:Optional[BitMap] = None) -> Tuple[Figure,Axes]:
        """
        Render a list of layers.
        :param layer_names:A list of layer names.
        :param alpha:The opacity of each layer.
        :param figsize:Size of the whole figure.
        :param tokens:Optional list of tokens to render. None means all tokens are rendered.
        :param bitmap:Optional BitMap object to render below the other map layers.
        :return:The matplotlib figure and axes of the rendered layers.
        """
        figsize1 =self._get_figsize(figsize)
        fig = plt.figure(figsize=figsize1)
        # fig = plt.figure(self._get_figsize(figsize))
        ax = fig.add_axes([0,0,1,1 / self.canvas_aspect_ratio])

        ax.set_xlim(self.canvas_min_x,self.canvas_max_x)
        ax.set_ylim(self.canvas_min_y,self.canvas_max_y)

        if bitmap is not None:
            if bitmap.bitmap_type == 'bitmap_mask':
                bitmap.render_mask_map(ax)
            elif bitmap.bitmap_type == 'bitmap_rgb':
                bitmap.render_rgb_map(ax)
            else:
                raise Exception('###Exception### 非法的 bitmap type:%s' % self.bitmap_type) # 自定义异常

        layer_names = list(set(layer_names))
        for layer_name in layer_names:
            self._render_layer(ax,layer_name,alpha,tokens)

        ax.legend()

        return fig,ax

    def render_map_patch(self,
                         box_coords:Tuple[float,float,float,float],
                         layer_names:List[str] = None,
                         alpha:float = 0.5,
                         figsize:Tuple[float,float] = (15,15),
                         render_egoposes_range:bool = True,
                         render_legend:bool = True,
                         bitmap:Optional[BitMap] = None,
                         fig:plt.figure = None) -> Tuple[Figure,Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        渲染一个矩形图框,指定矩形框的xy坐标范围.
        :param box_coords:The rectangular patch coordinates (x_min,y_min,x_max,y_max). 
                矩形框的局部 x ,y 坐标范围
        :param layer_names:All the non geometric layers that we want to render.
        :param alpha:The opacity of each layer.
                每一个图层的透明度.
        :param figsize:Size of the whole figure.
                整张fig的尺寸
        :param render_egoposes_range:是否渲染一个包围所有自身姿态( ego poses)的矩形.
        :param render_legend:Whether to render the legend of map layers.
        :param bitmap:Optional BitMap object to render below the other map layers.
        :return:The matplotlib figure and axes of the rendered layers.
        """
        x_min,y_min,x_max,y_max = box_coords

        # if layer_names is None:
        #     layer_names = self.map_api.non_geometric_layers

        if fig is None:
            fig = plt.figure(figsize=figsize)

        local_width = x_max - x_min
        local_height = y_max - y_min
        assert local_height > 0,'Error:Map patch has 0 height!'
        local_aspect_ratio = local_width / local_height #局部图框的横纵比例

        ax = fig.add_axes([0,0,1,1 / local_aspect_ratio]) #创建一个子图(axes)对象,并指定其位置和大小

        if bitmap is not None:
            if bitmap.bitmap_type == 'bitmap_mask':
                bitmap.render_mask_map(ax)
            elif self.bitmap_type == 'bitmap_rgb':
                bitmap.render_rgb_map(ax)
            else:
                raise Exception('###Exception### 非法的 bitmap type:%s' % self.bitmap_type) # 自定义异常
            
        for layer_name in layer_names:#渲染各图层
            self._render_layer(ax,layer_name,alpha)

        x_margin = np.minimum(local_width / 4,50) #xy追加边距
        y_margin = np.minimum(local_height / 4,10)
        ax.set_xlim(x_min - x_margin,x_max + x_margin)
        ax.set_ylim(y_min - y_margin,y_max + y_margin)

        if render_egoposes_range:
            ax.add_patch(Rectangle((x_min,y_min),local_width,local_height,fill=False,linestyle='-.',color='red',
                                   lw=3))
            ax.text(x_min + local_width / 100,y_min + local_height / 2,"%g m" % local_height,
                    color='red',fontsize=14,weight='bold')
            ax.text(x_min + local_width / 2,y_min + local_height / 100,"%g m" % local_width,
                    color='red',fontsize=14,weight='bold') #fig上添加文字标注

        if render_legend:
            ax.legend(frameon=True,loc='upper right')

        return fig,ax

    def extract_polygon(self,polygon_token:str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token. 用多边形标记构造一个形状良好的多边形对象.
        :param polygon_token:The token of the polygon record.
        :return:The polygon wrapped in a shapely Polygon object.
        """
        polygon_record = self.map_api.get('polygon',polygon_token)  # 得到了对应的多边形记录

        link_coords = [(self.map_api.get('node',token)['x'],self.map_api.get('node',token)['y'])
                           for token in polygon_record['link_node_tokens']]  # 多边形坐标列表

        return Polygon(link_coords)  # 返回多边形
    
    @staticmethod
    def get_patch_coord(patch_box_meter:Tuple[float,float,float,float],
                        patch_angle:float = 0.0) -> Polygon:
        """
        Convert patch_box_meter to shapely Polygon coordinates.
        :param Patch box :定义显示图框(补丁框)大小 [x_center,y_center,height,width],单位:米.
        :param patch_angle: 显示图框(补丁框)的方向角,单位:deg,正北为0 deg.
        :return:Box Polygon for patch_box_meter.
        """
        patch_x,patch_y,patch_h,patch_w = patch_box_meter

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min,y_min,x_max,y_max)
        patch = affinity.rotate(patch,patch_angle,origin=(patch_x,patch_y),use_radians=False)

        return patch

    def _get_figsize(self,figsize:Union[None,float,Tuple[float,float]]) -> Tuple[float,float]:
        """
        Utility function that scales the figure size by the map canvas size.按地图画布大小缩放图形大小的实用函数.
        If figsize is:
        - None      => Return default scale.
        - Scalar    => Scale canvas size.
        - Two-tuple => Use the specified figure size.
        :param figsize:The input figure size.
        :return:The outputs figure size.
        """
        # 将画布大小除以任意标量得到厘米范围.
        # canvas_size_pixel = np.array(self.map_api.canvas_edge_meter)[::-1] / 200
        canvas_size_pixel = np.array(self.map_api.canvas_edge_meter) / 200

        if figsize is None:
            return tuple(canvas_size_pixel)
        elif type(figsize) in [int,float]:
            return tuple(canvas_size_pixel * figsize)
        elif type(figsize) == tuple and len(figsize) == 2:
            return figsize
        else:
            raise Exception('Error:Invalid figsize:%s' % figsize)


    def _render_layer(self,ax:Axes,layer_name:str,alpha:float,tokens:List[str] = None) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax:The matplotlib axes where the layer will get rendered.
        :param layer_name:Name of the layer that we are interested in.
        :param alpha:The opacity of the layer to be rendered.
        :param tokens:Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            self._render_polygon_layer(ax,layer_name,alpha,tokens)
        elif layer_name in self.map_api.non_geometric_line_layers:
            # self._render_line_layer(ax,layer_name,alpha,tokens)#!待调试
            pass
        elif layer_name == "road_block":
            alpha = 1
            self._render_road_block_polygon_layer(ax,layer_name,alpha,tokens)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))


    def _render_polygon_layer(self,ax:Axes,layer_name:str,alpha:float,tokens:List[str] = None) -> None:
        """
        Renders an individual non-geometric polygon layer on an axis.
        :param ax:The matplotlib axes where the layer will get rendered.
        :param layer_name:Name of the layer that we are interested in.
        :param alpha:The opacity of the layer to be rendered.
        :param tokens:Optional list of tokens to render. None means all tokens are rendered.
        """ 
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        first_time = True
        records = getattr(self.map_api,layer_name) # 获得 某个图层'road'的内容,dict
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        else:
            for record in records:  # 一个token一个的画图
                polygon = self.map_api.extract_polygon(record['link_polygon_token'])  # 创建一个 Shapely 多边形对象

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None
                base_color = to_rgba(self.color_map[layer_name]) # Convert the hex to RGBA
                edge_color_with_alpha = (base_color[0],base_color[1],base_color[2],1)  # Set edge alpha to 1
                face_color_with_alpha = (base_color[0],base_color[1],base_color[2],0.1)  # Set face alpha to 0.1
                
                ax.add_patch(descartes.PolygonPatch(polygon,fc=face_color_with_alpha,ec=edge_color_with_alpha,label=label))
                # ax.add_patch(descartes.PolygonPatch(polygon,fc=self.color_map[layer_name],alpha=alpha,label=label) #!之前的代码
                
                
    def _render_road_block_polygon_layer(self,ax:Axes,layer_name:str,alpha:float,tokens:List[str] = None) -> None:
        """ 渲染road_block图层.是一个多边形,表示road is blocked
        """ 
        if layer_name != 'road_block':
            raise ValueError('{} is error,must road_block'.format(layer_name))

        first_time = True
        records = getattr(self.map_api,layer_name) # 获得 某个图层'road'的内容,dict
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        else:
            for record in records:#一个token一个的画图
                link_nodeblock_tokens=record['link_nodeblock_tokens']
                link_coords = [(self.map_api.get('node_block',token)['x'],self.map_api.get('node_block',token)['y'])
                           for token in link_nodeblock_tokens] #组成 road_block 多边形的角点
                polygon =  Polygon(link_coords)# 创建一个 Shapely 多边形对象

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None
                base_color = to_rgba(self.color_map[layer_name]) # Convert the hex to RGBA
                edge_color_with_alpha = (base_color[0],base_color[1],base_color[2],1)  # Set edge alpha to 1
                face_color_with_alpha = (base_color[0],base_color[1],base_color[2],0.5)  # Set face alpha to 0.1
                
                ax.add_patch(descartes.PolygonPatch(polygon,fc=face_color_with_alpha,ec=edge_color_with_alpha,label=label))            
   
                
    def render_base_path_centerlines(self,
                                    ax:Axes,
                                    alpha:float=0.5,
                                    resampling_rate:float = 0.2) -> None:
        """绘制参考路径,base_path.
        1)按照离散间隔绘制waypoints点;2)绘制所有dubinspose 
        
        Args:
            ax (Axes): x轴
            alpha (float, optional): 透明度. Defaults to 0.8.
            resampling_rate (float, optional): waypoints离散间隔采样. Defaults to 0.1.
                注:在这里,step = int(1 / resampling_rate)计算了降采样步长.
                如果resampling_rate是0.5,那么step值为2,所以我们每隔一个点取一个点来绘图.
                如果resampling_rate是0.25,那么step值为4,意味着我们每隔3个点取一个点.
                如果resampling_rate是0.2,那么step值为5,意味着我们每隔4个点取一个点.
                如果resampling_rate是0.1,那么step值为10,意味着我们每隔9个点取一个点.
        """
        for id_path,reference_path in enumerate(self.map_api.reference_path): 
            if reference_path['type'] != 'base_path':
                continue 
                    
            self.render_dubins_poses(ax=ax,alpha=0.4,dubinspose_tokens=reference_path['link_dubinspose_tokens'])
            if "waypoints" not in reference_path:   
                continue         
            waypoints = reference_path['waypoints']
            # Resampling
            step = int( 1 / resampling_rate )
            waypoints_resampling = np.array( waypoints[::step] )
            temp_color = self.color_map['connector_path']
            if len(waypoints_resampling) > 0:
                ax.plot(waypoints_resampling[:,0],waypoints_resampling[:,1],color=temp_color,linestyle='--',alpha=alpha,lw=1.2)
                
                    
    def render_connector_path_centerlines(self,
                                           ax:Axes,
                                           alpha:float=0.8,
                                           resampling_rate:float = 0.1) -> None:
        """绘制参考路径,connector_path.
        1)按照离散间隔绘制waypoints点;2)绘制所有dubinspose 
        
        Args:
            ax (Axes): x轴
            alpha (float, optional): 透明度. Defaults to 0.8.
            resampling_rate (float, optional): waypoints离散间隔采样. Defaults to 0.1.
                注:在这里,step = int(1 / resampling_rate)计算了降采样步长.
                如果resampling_rate是0.5,那么step值为2,所以我们每隔一个点取一个点来绘图.
                如果resampling_rate是0.25,那么step值为4,意味着我们每隔3个点取一个点.
                如果resampling_rate是0.2,那么step值为5,意味着我们每隔4个点取一个点.
                如果resampling_rate是0.1,那么step值为10,意味着我们每隔9个点取一个点.
        """
        
        for id_path,reference_path in enumerate(self.map_api.reference_path): 
            if reference_path['type'] != 'connector_path':
                continue
                         
            self.render_dubins_poses(ax=ax,alpha=0.4,dubinspose_tokens=reference_path['link_dubinspose_tokens'])  # 绘制dubinspose点及箭头
            if "waypoints" not in reference_path:   
                continue         
            waypoints = reference_path['waypoints']
            # Resampling
            step = int( 1 / resampling_rate )
            waypoints_resampling = np.array( waypoints[::step] )
            temp_color = self.color_map['connector_path']
            if len(waypoints_resampling) > 0:
                ax.plot(waypoints_resampling[:,0],waypoints_resampling[:,1],color=temp_color,linestyle='-.',alpha=alpha,lw=0.5)
   

            
    def render_dubins_poses(self,ax:Axes,alpha:float=0.5,dubinspose_tokens:List[str] = None) -> None:
        """绘制所有dubinspose.

        Args:
            ax (Axes): x轴
            alpha (float, optional): 透明度. Defaults to 0.5.
            dubinspose_tokens (List[str], optional): 杜宾斯点序列. Defaults to None.
        """
        for token_dubinspose in dubinspose_tokens:
            id_dubinspose = int( token_dubinspose.split('-')[1]) 
            if id_dubinspose > len(self.map_api.dubins_pose):
                continue # 语句被用来告诉 Python 跳过当前循环块中的剩余语句,然后继续进行下一轮循环.          
            x = self.map_api.dubins_pose[id_dubinspose]['x']
            y = self.map_api.dubins_pose[id_dubinspose]['y']
            yaw = self.map_api.dubins_pose[id_dubinspose]['yaw']
            temp_color = self.color_map['dubins_pose']
            # 绘制箭头dubinspose
            node_ux = x + 3*np.cos(yaw) 
            node_vy = y + 3*np.sin(yaw) 
            ax.arrow(x,y,node_ux-x,node_vy-y,color=temp_color,alpha=alpha,head_width=0.65,head_length=1.0)




if __name__ == "__main__":
    # 测试代码
    pass
     







 