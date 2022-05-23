from argparse import ArgumentParser
import json
from experiment import Experiment
from torch.multiprocessing import set_start_method
from multiprocessing import Process

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--run-id', type=str, help='Optional ID by which the current run should be saved')
    parser.add_argument('--config', type=str, metavar='CONFIG', help='Path to the config file',
                        default='config.json')
    parser.add_argument('--var', type=str, metavar='KEY=VALUE', action='append',
                        help='Key-value assignment for configuration variable - '
                             'will be updated in the current config file')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for Reproducibility')
    return parser.parse_args()


def parse_variable_assignment(assignments):
    vars = {}
    for ass in assignments:
        key, value = ass.split('=', 1)
        if 'e' in value or '.' in value:
            try:
                value = float(value)
            except ValueError:
                pass
        elif value.isdigit():
            value = int(value)
        vars[key] = value
    return vars


def run(config, run_id=None, seed=None):
    exp = Experiment(config, experiment_id=run_id)
    exp.run(seed=seed)


def main():
    options = create_parser()
    set_start_method('spawn')

    # load corresponding config
    config = json.load(open(options.config))

    if options.var is not None:
        updates = parse_variable_assignment(options.var)
        false_keys = [key for key in updates.keys() if key not in config]
        if len(false_keys):
            exc = ', '.join(false_keys)
            print(f"Added keys: {exc} to config...")
        config.update(updates)
        run_id = '_'.join([f'{k}={v}' for k, v in updates.items()])
    else:
        run_id = None

    p = Process(target=run, args=(config, run_id, options.seed))
    p.start()
    p.join()


if __name__ == '__main__':
    main()

