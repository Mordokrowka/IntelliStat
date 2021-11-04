import json
from functools import lru_cache, singledispatch
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from jsonschema import validate


@singledispatch
def wrap_namespace(ob):
    return ob


@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})


@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]


class BaseBuilder:
    def __init__(self):
        # Constant
        self.config_schema_file: Optional[Path] = None

    @lru_cache(maxsize=1)
    def load_config_schema(self, config_schema_file: Optional[Path] = None):
        with config_schema_file.open() as fp:
            return json.load(fp)

    @lru_cache(maxsize=1)
    def load_configuration(self, config_file: Path, config_schema_file: Optional[Path] = None):
        config_schema_file = config_schema_file if config_schema_file else self.config_schema_file
        with config_file.open() as fp:
            config_data = json.load(fp)
            validate(config_data, schema=self.load_config_schema(config_schema_file))
            return wrap_namespace(config_data)
