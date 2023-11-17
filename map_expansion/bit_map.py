# 内置库
import os
import time

# 第三方库
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
# from matplotlib.patches import Rectangle,Arrow
# from scipy.ndimage import zoom
from scipy.signal import convolve2d
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union, Any

Image.MAX_IMAGE_PIXELS = None
Axis = Any


# plt.style.use('seaborn-whitegrid')# Seaborn 库提供的一种基于白色网格背景的样式.


class BitMap:
    """BitMap类,用于读取、显示栅格地图等功能.
    矿区地图的信息包括以下栅格图层(rasterized):
    1)可行驶区域图.png,bitmap_mask.png
        - 江西-江铜:"jiangtong";
        - 广东-大排:"jiangtong";        
    使用与Google Maps/Earth 相同的WGS 84 Web Mercator (EPSG:3857);"CRS_name":"GCS_WGS_1984_UTM_Zone_50N".
    2) bitmap_rgb.png
        由航空制图转化为不带坐标信息的rgb栅格图,可直观显示整个区域全貌.
    """

    def __init__(self,
                 dataroot: str = '/home/czf/project_czf/20231010_onsite_mine/datasets',
                 location: str = "jiangxi_jiangtong",
                 bitmap_type: str = 'bitmap_mask',
                 is_transform_gray=False):
        """地图中的位图(栅格图)包括:
         - bitmap_mask:语义先验(可行驶区域)黑白图掩码(mask)图层, bitmap_mask.png文件 .
         - bitmap_rgb :将航空制图转化为PNG彩色图, bitmap_rgb.png文件.

        Args:
            dataroot (str,optional):datasets根目录. Defaults to '/home/czf/project_czf/20231010_onsite_mine/datasets'.
            location (str,optional):具体矿区位置的名称,当前有以下可选 :`jiangxi_jiangtong`,`guangdong_dapai`. Defaults to "jiangxi_jiangtong".
            bitmap_type (str,optional):位图类型,当前可选:`bitmap_mask`,`bitmap_rgb`..Defaults to 'bitmap_mask'.
            is_transform_gray (bool,optional):黑白图 OR 灰色图. Defaults to False.
        """

        self.dataroot = dataroot
        self.location = location
        self.bitmap_type = bitmap_type

        self.load_bitmap_info()
        self.image_ndarray = self.load_bitmap(is_transform_gray=is_transform_gray)

    def load_bitmap_info(self):
        """load bitmap information.
        """
        semantic_map_hashes = {
            "jiangxi_jiangtong": 'jiangxi_jiangtong_semantic_map',
            "guangdong_dapai": 'guangdong_dapai_semantic_map'}
        semantic_map_hash = semantic_map_hashes[self.location]
        dir_semantic_map = os.path.join(self.dataroot, "semantic_map", semantic_map_hash + '.json')
        with open(dir_semantic_map, 'r') as f:
            temp_semantic_map_info = json.load(f)
        self.bitmap_info = {}
        self.bitmap_info['bitmap_rgb_PNG'] = temp_semantic_map_info['bitmap_rgb_PNG']
        self.bitmap_info['bitmap_mask_PNG'] = temp_semantic_map_info['bitmap_mask_PNG']

    def load_bitmap(self, is_transform_gray=False) -> np.ndarray:
        """ 加载指定的 bitmap.
         
        Args:
            is_transform_gray (bool, optional): 黑白图 OR 灰色图. Defaults to False.

        Returns:
            np.ndarray: 栅格图数组(全局).
        """

        if self.bitmap_type == 'bitmap_mask':
            bitmap_hashes = {
                "jiangxi_jiangtong": 'jiangxi_jiangtong_bitmap_mask',
                "guangdong_dapai": 'guangdong_dapai_bitmap_mask'}
            bitmap_hash = bitmap_hashes[self.location]
            dir_bitmap = os.path.join(self.dataroot, 'bitmap', bitmap_hash + '.png')
            if os.path.exists(dir_bitmap):
                image = Image.open(dir_bitmap)
                image_ndarray = np.array(image.convert('1'))  # 转为 np.ndarray
                image_ndarray = np.flipud(image_ndarray)  # 翻转Y轴
                # image_ndarray = image_ndarray.max() - image_ndarray # 实现颜色反转的效果. 
            else:
                raise Exception('###Exception### %s的地图路径不存在 %s! Please check.' % (self.bitmap_type, dir_bitmap))

        elif self.bitmap_type == 'bitmap_rgb':
            bitmap_hashes = {
                "jiangxi_jiangtong": "jiangxi_jiangtong_bitmap_rgb",
                "guangdong_dapai": "guangdong_dapai_bitmap_rgb"}
            bitmap_hash = bitmap_hashes[self.location]
            dir_bitmap = os.path.join(self.dataroot, 'bitmap', bitmap_hash + '.png')
            if os.path.exists(dir_bitmap):
                if is_transform_gray == False:
                    image = Image.open(dir_bitmap)
                    image_ndarray = np.array(image.convert('RGB'))
                else:
                    image = Image.open(dir_bitmap)
                    image_ndarray = np.array(image.convert('L'))
            else:
                raise Exception('###Exception### %s的地图路径不存在 %s! Please check.' % (self.bitmap_type, dir_bitmap))

        else:
            raise Exception('###Exception### 非法的 bitmap type:%s' % self.bitmap_type)  # 自定义异常

        self.dir_bitmap = dir_bitmap

        return image_ndarray

    def load_bitmap_using_utm_local_range(self,
                                          utm_local_range: Tuple[float, float, float, float] = (0.0, 0.0, .1, .1),
                                          x_margin=10, y_margin=10):
        """
        根据局部坐标范围获取对应的局部区域的像素矩阵
        utm_local_range:The rectangular patch using utm local coordinates (x_min,y_min,x_max,y_max).
         

        Args:
            utm_local_range (Tuple[float,float,float,float], optional): The rectangular patch 
                using utm local coordinates (x_min,y_min,x_max,y_max).Defaults to (0.0,0.0,.1,.1).
            x_margin (int, optional): x外边缘尺寸. Defaults to 10m.
            y_margin (int, optional): y外边缘尺寸. Defaults to 10m.

        Returns:
            np.ndarray: 栅格图数组(局部).
        """

        self.bitmap_local_info = dict()
        self.bitmap_local_info["utm_local_range"] = utm_local_range
        self.bitmap_local_info["x_margin"] = x_margin
        self.bitmap_local_info["y_margin"] = y_margin
        # calculate pixel  
        px_min, py_min = self._utm_to_pixel(utm_local_range[0] - x_margin,
                                            utm_local_range[1] - y_margin)  # 注意像素坐标与UTM坐标在y轴上的方向映射是正确的
        px_max, py_max = self._utm_to_pixel(utm_local_range[2] + x_margin, utm_local_range[3] + y_margin)
        self.bitmap_local_info["pixel_local_range"] = (px_min, py_min, px_max, py_max)

        # 修正像素尺寸为方形
        px_diff = px_max - px_min
        py_diff = py_max - py_min
        max_diff = max(px_diff, py_diff)
        px_max = px_min + max_diff
        py_max = py_min + max_diff

        # 使用像素坐标从self.image_ndarray中切片出局部矩阵
        self.image_ndarray_local = self.image_ndarray[py_min:py_max, px_min:px_max]

    def _utm_to_pixel(self, x, y):
        """将UTM坐标(x,y)转换为像素坐标(pixel_x,pixel_y).
        """

        x_range = self.bitmap_info['bitmap_mask_PNG']['UTM_info']['local_x_range']
        y_range = self.bitmap_info['bitmap_mask_PNG']['UTM_info']['local_y_range']
        scale = self.bitmap_info['bitmap_mask_PNG']['scale_PixelPerMeter']

        # 转换坐标
        pixel_x = int((x - x_range[0]) * scale)
        pixel_y = int((y - y_range[0]) * scale)  # 注意y轴已经反过来了

        return pixel_x, pixel_y  # 返回的是像素点的索引（右下或者左上）

    def render_mask_map_using_image_ndarray_local(self, ax: Axis = None, window_size=4, gray_flag=True):
        """ 渲染可行驶区域二值化栅格图(局部);
        注:黑白图 和 灰色图 两种.        
         
        Args:
            ax (Axis,optional):坐标轴ax. Defaults to None.
            window_size (int,optional):降采样窗口大小,使用均值滤波器. Defaults to 4.
            gray_flag (bool,optional):黑白图 OR 灰色图. Defaults to True.
        """
        if self.bitmap_type != 'bitmap_mask':
            raise ValueError('Error:render_mask_map()函数只有使用semantic_prior图初始化bitmap才可以调用!')

        if ax is None:
            ax = plt.subplot()

        # 滤波+降采样
        image_ndarray_local = self.downsample_with_filter(self.image_ndarray_local, window_size)

        # 根据降采样的比率调整UTM的范围
        x_min = self.bitmap_local_info['utm_local_range'][0] - self.bitmap_local_info['x_margin']
        y_min = self.bitmap_local_info['utm_local_range'][1] - self.bitmap_local_info['y_margin']
        x_max = self.bitmap_local_info['utm_local_range'][2] + self.bitmap_local_info['x_margin']
        y_max = self.bitmap_local_info['utm_local_range'][3] + self.bitmap_local_info['y_margin']

        if gray_flag == True:  # 黑白图
            ax.imshow(image_ndarray_local, extent=(x_min, x_max, y_min, y_max), cmap='gray', origin='lower')
        else:  # 创建从灰到白的自定义 colormap
            colors = [(0.5, 0.5, 0.5), (1, 1, 1)]  # R -> G -> B
            n_bins = [3]  # Discretizes the interpolation into bins
            cmap_name = 'custom_div_cmap'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
            ax.imshow(image_ndarray_local, extent=(x_min, x_max, y_min, y_max), cmap=cm, origin='lower')

    @staticmethod
    def downsample_with_filter(image: np.ndarray = None, window_size: int = 5) -> np.ndarray:
        """均值滤波器对图像进行降采样.

        Args:
            image (np.ndarray, optional): 原图像数组. Defaults to None.
            window_size (int, optional): 滤波窗口尺寸. Defaults to 5.

        Returns:
            np.ndarray: 滤波后图像数组.
        """

        # 定义一个均值滤波器
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
        # 使用滤波器对图像进行滤波
        filtered_image = convolve2d(image.astype(float), kernel, mode='same', boundary='wrap')
        # 降采样
        downsampled = filtered_image[::window_size, ::window_size]
        # 二值化
        downsampled_binary = (downsampled > 0.5).astype(np.bool)

        return downsampled_binary

    def render_mask_map(self, ax: Axis = None, gray_flag: bool = True):
        """ 渲染可行驶区域二值化栅格图(全局图像),semantic_prior.
        注1:整个图都会被渲染,只是显示范围由其它调用函数决定;注2:黑白图 和 灰色图 两种.  

        Args:
            ax (Axis,optional):可选参数,用于指定渲染图像的轴(坐标轴). Defaults to None.
            gray_flag (bool,optional):黑白图 OR 灰色图. Defaults to True.
        """

        assert self.bitmap_type == 'bitmap_mask', 'Error:render_mask_map()函数只有使用bitmap_mask图初始化bitmap才可以调用!'

        if ax is None:
            ax = plt.subplot()

        # 当前map的实际地理尺寸大小,(宽,高)米 . meters (width,height).
        x_min, x_max = self.UTM_local_x_range
        y_min, y_max = self.UTM_local_y_range
        ax.imshow(self.image_ndarray, extent=[x_min, x_max, y_min, y_max], cmap='gray', origin='lower')

        if gray_flag == True:  # 黑白图
            ax.imshow(self.image_ndarray, extent=[x_min, x_max, y_min, y_max], cmap='gray', origin='lower')
        else:  # 创建从灰到白的自定义 colormap
            colors = [(0.5, 0.5, 0.5), (1, 1, 1)]  # R -> G -> B
            n_bins = [3]  # Discretizes the interpolation into bins
            cmap_name = 'custom_div_cmap'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
            ax.imshow(self.image_ndarray, extent=[x_min, x_max, y_min, y_max], cmap=cm, origin='lower')

    def render_rgb_map(self, ax: Axis = None):
        """渲染栅格图中的 bitmap_rgb.png
 
        Args:
            ax (Axis,optional):可选参数,用于指定渲染图像的轴(坐标轴). Defaults to None.
        """

        assert self.bitmap_type == 'bitmap_rgb', 'Error:render_rgb_map()函数只有使用GeoTIFF_2_PNG图初始化bitmap才可以调用!'

        if ax is None:
            ax = plt.subplot()
        x, y = self.canvas_edge_meter
        if len(self.image_ndarray.shape) == 2:
            ax.imshow(self.image_ndarray, extent=[0, x, 0, y], cmap='gray')
        else:
            ax.imshow(self.image_ndarray, extent=[0, x, 0, y])
