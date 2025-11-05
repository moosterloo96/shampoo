import json

class LoadConfig:

    def load_config(self, path="src/model/"):

        with open(path + "config.json") as f:
            config = json.load(f)

        return config

    def complement_config(self, manual_config, path="src/model/"):

        with open(path + "config.json") as f:
            default_config = json.load(f)

        autofilled_config = manual_config
        for key, value in default_config.items():
            if key not in manual_config:
                autofilled_config[key] = value

        return autofilled_config