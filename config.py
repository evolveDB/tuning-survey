import configparser
import json
class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d

db_config=None
knob_config=None
cfp=DictParser()
cfp.read("../config.ini", encoding="utf-8")
config_dict=cfp.read_dict()
db_config=config_dict["database"]
knob_config=config_dict["Non-restart Knobs"]
for key in knob_config:
    knob_config[key] = json.loads(str(knob_config[key]).replace("\'", "\""))
    
    