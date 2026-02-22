
import argparse

def parse_experiment_args():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--profiles_json",
            type=str,
            help="JSON string"
        )

    parser.add_argument(
            "--N",
            type=int,
            nargs="+",
            default=[50],
            help="Set the value of N (default: 50)"
        )

    parser.add_argument(
            "--log_scale",
            action="store_true",
            default=False,
            help="Use logarithmic scale"
        )

    return parser.parse_args()
