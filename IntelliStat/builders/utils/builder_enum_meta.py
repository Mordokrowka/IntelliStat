from enum import EnumMeta


class BuilderEnumMeta(EnumMeta):
    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"`index` {index} is not an int")
        if index >= super().__len__():
            raise KeyError("Key error")

        return list(self)[index]
