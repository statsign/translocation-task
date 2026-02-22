
from compare_profiles import CompareProfiles
from optimizers.multi_optimizer import MultipleOptimizer


class ExperimentSeries:
    def __init__(self, solver, profile_sets, N_values=None, log_scale=False):
        """
        Run a series of experiments with different profile sets

        Args:
            solver: FokkerPlanckSolver instance
            profile_sets: List of profile sets
            N_values: List of N values to use in experiments (if None, use default N=50)
            log_scale: Whether to use log scale
        """
        self.solver = solver
        self.profile_sets = profile_sets
        self.N_values = N_values if N_values else [50]
        self.log_scale = log_scale

    def run_experiments(self, max_iter=5, initial_points=50):
        """
        Run a series of experiments

        Args:
            max_iter: Maximum number of iterations for optimization
            initial_points: Number of initial points for optimization
        """
        results = {}

        for exp_id, profiles in enumerate(self.profile_sets):
            print(f"\n{'='*50}")
            print(f"Starting Experiment {exp_id}")
            print(f"{'='*50}")

            exp_results = {}

            for N in self.N_values:
                print(f"\nRunning experiment {exp_id} with N={N}")

                # Initialize comparison
                compare = CompareProfiles(
                    profiles=profiles, log_scale=self.log_scale, experiment_id=exp_id)

                # Initial results
                fp_results = compare.run_multiple_simulations(N)

                # Initialize optimizer
                optimizer = MultipleOptimizer(
                    self.solver, N_ref=N, profiles=profiles, log_scale=self.log_scale, experiment_id=exp_id)

                # Compute reference models
                success = optimizer.compute_reference_models()

                if success:
                    # Run optimization
                    opt_results = optimizer.run_multiple_opt(
                        max_iter=max_iter, initial_points=initial_points)

                    exp_results[N] = {
                        'simulation_results': fp_results,
                        'optimization_results': opt_results
                    }
                else:
                    print(
                        f"Failed to compute reference models for N={N}, experiment {exp_id}")

            results[f'experiment_{exp_id}'] = exp_results

            print(f"\n{'='*50}")
            print(f"Experiment {exp_id} completed")
            print(f"{'='*50}")

            self.create_summary_report(exp_id, results)

        return results