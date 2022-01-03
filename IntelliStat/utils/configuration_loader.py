import json
from functools import singledispatch
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union


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


def load_configuration(config_file_path: Union[Path, str],
                       config_schema_file_path: Union[Path, str]) -> SimpleNamespace:
    """Loads json configuration file and validates against json schema.

    :param config_file_path: path to config file
    :param config_schema_file_path: path to json schema file
    :return: configuration data
    """
    with open(config_file_path, encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    with open(config_schema_file_path) as config_schema_file:
        config_schema = json.load(config_schema_file)

    validate(config_data, schema=config_schema)
    return wrap_namespace(config_data)
