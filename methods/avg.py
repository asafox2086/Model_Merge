from .base import MergeMethod
from utils.state_dict import average_state_dicts


class AvgMerge(MergeMethod):
    name = 'avg'

    def merge(self, state_dicts, weights):
        return average_state_dicts(state_dicts, weights)


def merge_avg(state_dicts, weights):
    return AvgMerge().merge(state_dicts, weights)
