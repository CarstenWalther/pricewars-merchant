from typing import Sequence


class PricewarsObject:

    def to_dict(self):
        return vars(self)

    def __repr__(self):
        return repr(vars(self))

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_list(cls, l: Sequence):
        return [cls.from_dict(e) for e in l]
