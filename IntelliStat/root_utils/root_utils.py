import os
from pathlib import Path
from pprint import pprint
from typing import Union

import numpy as np
import uproot


def save_data_to_root(root_file_path: Union[str, os.PathLike], item_name: str, branches: dict, update: bool = False) -> bool:
    open_root = uproot.recreate if not update else uproot.update
    with open_root(root_file_path) as root_file:
        root_file[item_name] = branches
    return True


if __name__ == '__main__':
    test_root_file = "D:\\Documents\\studia\\semestr7\\inzynierka\\IntelliStat\\examples\\classifiers\\multi_shape_classifier\\test.root"

    with uproot.open(test_root_file) as file:
        print(file.keys())
        print(file['12000'].show())
        for branch in file['12000']:
            pprint(branch.array(library='np'))
        # root_data = file['tree']
        # for branch in root_data:
        #     pprint(branch.array(library="pd"))
