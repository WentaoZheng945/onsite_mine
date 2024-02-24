import os
import sys
from copy import deepcopy

class ScenarioOrganizer():
    def __init__(self):
        self.test_mode = ""
        self.scenario_list = []
        self.test_num = 0
        self.scaler = None
 
    def load(self,dir_inputs:str,dir_outputs:str) -> None:
        """读取配置文件,按照配置文件(py格式)准备场景内容
        """
        self._check_output_dir(dir_outputs)
        self.scenario_list = []
        sys.path.append(dir_inputs)
        # dir_test_conf = os.path.abspath(dir_inputs,"test_conf.py")
        from test_conf import config   
        self.config = config
        self.test_mode = config['test_settings']['mode']  # replay

        self.config['file_info'] = {
            'dir_inputs':dir_inputs,
            'dir_outputs':dir_outputs,
            'dir_scenarios':os.path.join(dir_inputs,'Scenarios'),
            'dir_maps':os.path.join(dir_inputs,'Maps'),
            'location':"jiangxi_jiangtong",# default
        }

        self.config['test_settings'].setdefault('visualize',False)
         
        # 如果测试模式 == 'replay',读取文件夹下所有待测场景,存在列表中
        if self.test_mode == 'replay': # 判断测试模式是否为replay
            self.config['test_settings'].setdefault('skip_exist_scene',False)
            for item in os.listdir(self.config['file_info']['dir_scenarios']):
                if self.config['test_settings']['skip_exist_scene'] and os.path.exists(os.path.join(dir_outputs,item+'_result.csv')):
                    continue
                # dir_scene_file = self.config['file_info']['dir_ScenariosResultes'] + "/" + item
                dir_scene_file = os.path.join(self.config['file_info']['dir_scenarios'], item)

                sce = self.config.copy()
                scene_name_1 = item[9:-5]
                sce['data'] = {
                    'scene_name':scene_name_1,
                    'dir_scene_file':dir_scene_file
                }
                location_temp = sce['data']['scene_name'].split("_")[0]  # jiangtong/dapai
                if location_temp == "jiangtong":
                    sce['file_info']['location'] = "jiangxi_jiangtong"
                elif location_temp == "dapai":
                    sce['file_info']['location'] = "guangdong_dapai"
                elif location_temp == "hailuo":
                    sce['file_info']['location'] = "anhui_hailuo"
                else:
                    raise Exception('###Exception### 地图location 错误!')
                # 将场景加入列表中
                self.scenario_list.append (deepcopy(sce))
                
            self.test_num = len(self.scenario_list)

    def next(self):
        """
        给出下一个待测场景与测试模式,如果没有场景了,则待测场景名称为None
        """
        # 首先判断测试的模式,replay模式和adaptive模式不一样
        if self.test_mode == 'replay': # 如果是回放测试
            if self.scenario_list: # 首先判断列表是否为空,如果列表不为空,则取场景;否则,输出None
                # 列表不为空,输出0号场景,且将其从列表中删除(通过pop函数实现)
                scenario_to_test = self.scenario_list.pop(0)
            else:
                # 列表为空,输出None
                scenario_to_test = None
        else:
            scenario_to_test = None
        return scenario_to_test

    def add_result(self,concrete_scenario:dict,res:float) -> None:
        # 判断测试模式,如果是replay,则忽略测试结果
        if self.test_mode == 'replay':
            return

    def _check_output_dir(self,dir_outputs:str) -> None:
        """检查输出文件夹是否存在,如果不存在,则创建

        """
        if not os.path.exists(dir_outputs):
            os.makedirs(dir_outputs)


if __name__ == "__main__":
    pass
