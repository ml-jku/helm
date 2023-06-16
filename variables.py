procgen_envs = ["caveflyer", "dodgeball", "miner", "jumper", "maze", "heist"]

avalon_templates = [
    'a screenshot of {}',
    'a screenshot of a {}',
    'a screenshot of many {}',
    'a biome containing {}',
    'a biome containing a {}',
    'a biome containing many {}',
    'a biome full of {}'
]

dmlab_templates = [
    "a render of {}",
    "a render of a {}",
    "a screenshot of a {}",
    "a screenshot of {}",
    "a screen showing {}",
    "a screen showing a {}"
]

dmlab_envs = {
    'psychlab_continuous_recognition': 'contributed/psychlab/continuous_recognition'
}

_ACTION_MAP = {
    0: (0, 0, 0, 1, 0, 0, 0),
    1: (0, 0, 0, -1, 0, 0, 0),
    2: (0, 0, -1, 0, 0, 0, 0),
    3: (0, 0, 1, 0, 0, 0, 0),
    4: (-10, 0, 0, 0, 0, 0, 0),
    5: (10, 0, 0, 0, 0, 0, 0),
    6: (-60, 0, 0, 0, 0, 0, 0),
    7: (60, 0, 0, 0, 0, 0, 0),
    8: (0, 10, 0, 0, 0, 0, 0),
    9: (0, -10, 0, 0, 0, 0, 0),
    10: (-10, 0, 0, 1, 0, 0, 0),
    11: (10, 0, 0, 1, 0, 0, 0),
    12: (-60, 0, 0, 1, 0, 0, 0),
    13: (60, 0, 0, 1, 0, 0, 0),
    14: (0, 0, 0, 0, 1, 0, 0),
}

ACTIONS = [
    [-20, 0, 0, 0, 0, 0, 0],
    [20, 0, 0, 0, 0, 0, 0],
    [0, 10, 0, 0, 0, 0, 0],
    [0, -10, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
]