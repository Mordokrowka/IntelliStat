import json
from functools import singledispatch
from pathlib import Path
from typing import List, Optional
from types import SimpleNamespace

from jsonschema import validate


@singledispatch
def wrap_namespace(ob) -> SimpleNamespace:
    return ob


@wrap_namespace.register(dict)
def _wrap_dict(ob) -> SimpleNamespace:
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})


@wrap_namespace.register(list)
def _wrap_list(ob) -> List[SimpleNamespace]:
    return [wrap_namespace(v) for v in ob]


class BaseBuilder:
    def __init__(self):
        # Constant
        self.config_schema_file: Optional[Path] = None

    @staticmethod
    def load_config_schema(config_schema_file: Path):
        with open(config_schema_file) as fp:
            return json.load(fp)

    def load_configuration(self, config_file: Path, config_schema_file: Optional[Path] = None) -> SimpleNamespace:
        config_schema_file = config_schema_file if config_schema_file else self.config_schema_file
        with open(config_file) as fp:
            config_data = json.load(fp)
            validate(config_data, schema=self.load_config_schema(config_schema_file))
            return wrap_namespace(config_data)
