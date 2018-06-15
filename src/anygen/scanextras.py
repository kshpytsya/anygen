import collections as _collections
import inspect as _inspect


def scan(py_dict):
    result = _collections.defaultdict(dict)

    for name, val in py_dict.items():
        if name.startswith('_'):
            continue

        split_name = name.split('_', 1)
        if len(split_name) == 1:
            continue

        name_head, name_tail = split_name
        result[name_head][name_tail] = val

    return result


def scanme():
    return scan(_inspect.stack(context=0)[1].frame.f_globals)
