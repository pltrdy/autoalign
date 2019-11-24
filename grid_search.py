
def subgrid(params, subgrids):
    """
        Grid search with sub-sets of parameters
        each (key, values) in subgrids makes a parameter of the grid as:
        params[key] = grid_search(values)

    """
    for k, v in subgrids.items():
        params[k] = grid_search_params(v)

    return params


def grid_search_params(params):
    """ 
        Given a dict of parameters, return a list of dicts
    """
    params_list = []
    for ip, (param, options) in enumerate(params.items()):
        if ip == 0:
            params_list += [{param: v} for v in options]

            continue

        _params_list = [dict(_) for _ in params_list]

        for iv, value in enumerate(options):
            if iv > 0 or ip == 0:
                to_update = [dict(_) for _ in _params_list]
                params_list += to_update
            else:
                to_update = params_list

            for param_set in to_update:
                param_set[param] = value
    return params_list
