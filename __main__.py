from solver.fokker_planck import FokkerPlanckSolver
from experiments.experiments_series import ExperimentSeries
from utils.cli import parse_experiment_args

def main():
    args = parse_experiment_args()

    solver = FokkerPlanckSolver()
    experiment = ExperimentSeries(
        solver,
        profile_sets=args.profile_sets,
        N_values=args.N,
        log_scale=args.log_scale
    )
    experiment.run_experiments(
        max_iter=args.max_iter,
        initial_points=args.init_points,
        loss_mode=args.loss_mode
    )

if __name__ == "__main__":
    main()