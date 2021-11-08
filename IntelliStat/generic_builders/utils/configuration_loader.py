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


def load_configuration(config_file: Union[Path, str], config_schema_file: Union[Path, str]) -> SimpleNamespace:
    """Loads json configuration file and validates against json schema.

    :param config_file: path to config file
    :param config_schema_file: path to json schema file
    :return: configuration data
    """
    with open(config_file) as fp:
        config_data = json.load(fp)
        with open(config_schema_file) as fp:
            schema = json.load(fp)

        validate(config_data, schema=schema)
        return wrap_namespace(config_data)
