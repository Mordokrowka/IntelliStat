import os

from pprint import pprint
from typing import Union

import uproot


def save_data_to_root(root_file_path: Union[str, os.PathLike], item_name: str,
                      branches: dict, update: bool = False) -> bool:
    open_root = uproot.recreate if not update else uproot.update
    with open_root(root_file_path) as root_file:
        root_file[item_name] = branches
    return True


if __name__ == '__main__':
    test_root_file = "/examples/classifiers/multi_shape_classifier/multi_shape.root"

    with uproot.open(test_root_file) as file:
        print(file.keys())
        print(file['10200'].show())
        for branch in file['10200']:
            pprint(branch.array(library='np'))

