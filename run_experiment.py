from solver.fokker_planck import FokkerPlanckSolver
from experiments.experiments_series import ExperimentSeries
import traceback
import json
from utils.cli import parse_experiment_args


def main():

    args = parse_experiment_args()

    # Initialize solver
    solver = FokkerPlanckSolver()

    try:
        profile_sets = json.loads(args.profiles_json)

        # Create an experiment series with the profile sets from JSON
        experiment_series = ExperimentSeries(
            solver=solver,
            profile_sets=profile_sets,
            N_values=args.N,
            log_scale=args.log_scale
        )

        # Run experiments
        results = experiment_series.run_experiments()

        print("All experiments completed successfully!")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON profiles: {e}")
    except Exception as e:
        print(f"Error running experiments: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()