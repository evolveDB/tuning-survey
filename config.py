import configparser
import json
import numpy as np


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


class Logger():
    def __init__(self, path) -> None:
        self.path = path

    def write(self, string: str):
        f = open(self.path, 'a')
        f.write(string)
        f.close()


db_config = None
knob_config = None
remote_config = None
cfp = DictParser()
cfp.read("../my_config.ini", encoding="utf-8")
config_dict = cfp.read_dict()
db_config = config_dict["database"]
remote_config = config_dict["remote-access"]
knob_config = config_dict["Non-restart Knobs"]
for key in knob_config:
    knob_config[key] = json.loads(str(knob_config[key]).replace("\'", "\""))


def modifyKnobConfig(knob_info, user_define_config):
    knob_names = []
    knob_min = []
    knob_max = []
    knob_length = []
    knob_granularity = []
    knob_type = []
    for key in knob_info:
        knob_names.append(key)
        if "min" in user_define_config[key]:
            knob_min.append(user_define_config[key]["min"])
        else:
            knob_min.append(knob_info[key]['min'])

        if "max" in user_define_config[key]:
            knob_max.append(user_define_config[key]["max"])
        else:
            knob_max.append(knob_info[key]['max'])

        if "length" in user_define_config[key]:
            knob_length.append(user_define_config[key]["length"])

        if "granularity" in user_define_config[key]:
            knob_granularity.append(user_define_config[key]["granularity"])
        else:
            knob_granularity.append(knob_info[key]['granularity'])
        knob_type.append(knob_info[key]["type"])
    knob_min = np.array(knob_min)
    knob_max = np.array(knob_max)
    knob_length = np.array(knob_length)
    knob_granularity = np.array(knob_granularity)
    return knob_names, knob_min, knob_max, knob_granularity, knob_type, knob_length