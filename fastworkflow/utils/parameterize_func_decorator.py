from functools import wraps
from itertools import product

from fastworkflow.session import Session


def parameterize(**param_dict):
    def decorator(func):
        @wraps(func)
        def wrapper(session: Session):
            param_names = list(param_dict.keys())
            param_values = list(param_dict.values())
            combinations = list(product(*param_values))

            results = []
            for combo in combinations:
                params = dict(zip(param_names, combo))
                results.extend(func(session, **params))

            return results

        return wrapper

    return decorator
