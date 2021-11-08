import json
from functools import singledispatch
from pathlib import Path
from typing import List, Union
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


def load_configuration(config_file: Union[Path, str], config_schema_file: Union[Path, str]) -> SimpleNamespace:
    with open(config_file) as fp:
        config_data = json.load(fp)
        validate(config_data, schema=load_config_schema(config_schema_file))
        return wrap_namespace(config_data)


def load_config_schema(config_schema_file: Union[Path, str]):
    with open(config_schema_file) as fp:
        return json.load(fp)
