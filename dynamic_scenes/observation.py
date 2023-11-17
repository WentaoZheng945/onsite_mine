class Observation():
    """
    vehicle_info={
        "ego":{
            "x":,
            "y":,
            "yaw_rad":,
            "v_mps":,
            "yawrate_radps":,
            "acc_mpss":,
            "shape":{
				"vehicle_type":"MineTruck_NTE200",
				"length":13.4,
				"width":6.7,
				"height":6.9,
				"min_turn_radius":14.2,
				"locationPoint2Head":9.8,
				"locationPoint2Rear":3.8
			}
        },
        "0":{...},
        ...
    }
    test_setting = {
            "t":,
            "dt":,
            "max_t",
            "goal":{
                "x":[-1,-1,-1,-1],
                "y":[-1,-1,-1,-1]
            },
            "end":
        }
    """

    def __init__(self):
        self.vehicle_info = {
            "ego":{
                "x":-1,
                "y":-1,
                "yaw_rad":-1,
                "v_mps":-1,
                "yawrate_radps":-1,
                "acc_mpss":-1,
                "shape":{
                    "vehicle_type":"MineTruck_NTE200",
                    "length":13.4,
                    "width":6.7,
                    "height":6.9,
                    "min_turn_radius":14.2,
                    "locationPoint2Head":9.8,
                    "locationPoint2Rear":3.8
			    }
            },
        }
        self.hdmaps = {}
        self.test_setting = {
            "scenario_name":"name",
            "scenario_type":"replay",
            "t":0.0,
            "dt":0.1,
            "max_t":-1,
            "goal":{
                "x":[-1,-1,-1,-1],
                "y":[-1,-1,-1,-1]
            },
            "end":-1,
            "x_min":None,
            "x_max":None,
            "y_min":None,
            "y_max":None,
            "start_ego_info":None  
        }
        
    def format(self):
        return {
            "vehicle_info":self.vehicle_info,
            "test_setting":self.test_setting,
            "hdmaps_info":self.hdmaps,
        }

if __name__ == "__main__":
    pass
