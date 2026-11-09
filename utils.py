# copyright 2023 omnisafe team. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
# ==============================================================================
import hashlib
import itertools
import json
import sys
from typing import Any

import numpy as np
from rich.console import Console
import scipy


def assert_with_exit(condition: bool, msg: str) -> None:
    """assert with message.

    examples:
        >>> assert_with_exit(1 == 2, '1 must equal to 2')
        assertionerror: 1 must equal to 2

    args:
        condition (bool): condition to be checked.
        msg (str): message to be printed.

    raises:
        assertionerror: if the condition is not satisfied.
    """
    try:
        assert condition
    except AssertionError:
        console = Console()
        console.print("error: " + msg, style="bold red")
        sys.exit(1)


def hash_string(string: str) -> str:
    """This function is used to generate the folder name.

    Args:
        string (str): string to be hashed.

    Returns:
        The hashed string.
    """
    salt = b"\xf8\x99/\xe4\xe6J\xd8d\x1a\x9b\x8b\x98\xa2\x1d\xff3*^\\\xb1\xc1:e\x11M=PW\x03\xa5\\h"
    # convert string to bytes and add salt
    salted_string = salt + string.encode("utf-8")
    # use sha256 to hash
    hash_object = hashlib.sha256(salted_string)
    # get the hex digest
    return hash_object.hexdigest()


def recursive_dict2json(dict_obj: dict[str, Any]) -> str:
    """This function is used to recursively convert the dict to json.

    Args:
        dict_obj (dict[str, Any]): dict to be converted.

    Returns:
        The converted json string.

    Raises:
        AssertionError: If the instance of the input is not ``dict``.
    """
    assert isinstance(dict_obj, dict), "Input must be a dict."
    flat_dict: dict[str, Any] = {}

    def _flatten_dict(dict_obj: dict[str, Any] | Any, path: str = "") -> None:
        if isinstance(dict_obj, dict):
            for key, value in dict_obj.items():
                _flatten_dict(value, path + key + ":")
        else:
            flat_dict[path[:-1]] = dict_obj

    _flatten_dict(dict_obj)
    return json.dumps(flat_dict, sort_keys=True).replace('"', "'")


def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> None:
    """Updater of multi-level dictionary.

    Args:
        total_dict (dict[str, Any]): The total dictionary.
        item_dict (dict[str, Any]): The item dictionary.

    Examples:
        >>> total_dict = {'a': {'b': 1, 'c': 2}}
        >>> item_dict = {'a': {'b': 3, 'd': 4}}
        >>> update_dict(total_dict, item_dict)
        >>> total_dict
        {'a': {'b': 3, 'c': 2, 'd': 4}}
    """
    for idd in item_dict:
        total_value = total_dict.get(idd)
        item_value = item_dict.get(idd)

        if total_value is None:
            total_dict.update({idd: item_value})
        elif isinstance(item_value, dict):
            update_dict(total_value, item_value)
            total_dict.update({idd: total_value})
        else:
            total_value = item_value
            total_dict.update({idd: total_value})


def dict_permutations(input_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all possible combinations of the values in a dictionary.

    Takes a dictionary with string keys and list values, and returns a dictionary with all
    possible combinations of the lists as values for each key.

    Args:
        input_dict (dict[str, Any]): The input dictionary.

    Returns:
        The list of all possible combinations of the values in a dictionary.
    """
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    value_combinations = list(itertools.product(*values))

    result = []
    for combination in value_combinations:
        new_dict = {}
        for i, item in enumerate(keys):
            new_dict[item] = [combination[i]]
        result.append(new_dict)

    return result


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[1]
    mean = np.mean(a, axis=0)
    if a.shape[0] > 1:
        se = scipy.stats.sem(a, axis=0)
        ci = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean, ci
    else:
        return mean, [0] * mean.shape[0]
