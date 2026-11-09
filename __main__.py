import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fokker_planck.solver import FokkerPlanckSolver
from experiments.experiment_series import ExperimentSeries
from src.utils.cli import parse_experiment_args, load_profile_sets
from src.utils.path_utils import get_json_path

from src.utils.logger_setup import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    args = parse_experiment_args()

    solver = FokkerPlanckSolver()
    experiment = ExperimentSeries(
        solver,
        profile_sets=load_profile_sets(get_json_path()),
        N_values=args.N
    )
    experiment.run_experiments(
        max_iter=args.max_iter,
        initial_points=args.init_points
    )

if __name__ == "__main__":
    main()