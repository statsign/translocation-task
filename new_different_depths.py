from main import FokkerPlanckSolver
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt
import traceback
import os
import glob
from cycler import cycler
import json
import argparse
import pandas as pd

job_id = os.getenv('SLURM_JOB_ID', 'local')

data_folder = os.path.join("/data1/val2204", "data", f"job_{job_id}")
images_folder = os.path.join("/data1/val2204", "images", f"job_{job_id}")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

png_files = glob.glob(os.path.join(images_folder, '*.png'))

for file in png_files:
    os.remove(file)


class CompareProfiles:
    def __init__(self, profiles, log_scale=False, experiment_id=0):

        # Initialize solver
        self.solver = FokkerPlanckSolver()
        self.profiles = profiles
        self.log_scale = log_scale
        self.experiment_id = experiment_id

    def run_multiple_simulations(self, N):
        self.plot_profiles(N)

        results = []
        for profile in self.profiles:
            result = self.solver.run_fp(
                N, profile_type=profile['type'], params=profile['params'],
                name=f"{profile['name']}_exp{self.experiment_id}", 
                log_scale=self.log_scale, exp_id=self.experiment_id)
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['profile'] = profile
                results.append(result)

        self.plot_multiple_pdf(results, N)
        self.compare_multiple_pdf(results, N)

        return results

    def plot_profiles(self, N):

        fig, axes = plt.subplots(3, 1)
        axes = axes.flatten()
        for i, profile in enumerate(self.profiles):
            # Generate the profile first
            result = self.solver.gen_profile(
                N, profile['type'], profile['params'])
            zn = result['dt']
            F = result['F']

            axes[i].plot(zn, F, label=f"{profile['label']}")
            axes[i].legend(loc='best')

        plt.tight_layout()
        imgname = f"profiles_exp{self.experiment_id}_{N}"
        img_path = os.path.join(images_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def plot_multiple_pdf(self, results, N):
        if not results:
            print("No data to display")
            return
        fig, axes = plt.subplots(3, 1)
        axes = axes.flatten()

        for i, result in enumerate(results):
            filename = f"{result['name']}_exp{self.experiment_id}_N{N}_out.npz"
            path = os.path.join(data_folder, filename)
            try:
                with np.load(path) as data:
                    dt = data['dt']
                    total = data['ptotal']
                    if len(total.shape) > 1:
                        total = total.flatten()
                    success = data['pTimeN']
                    failure = data['pTime0']

                axes[i].plot(dt, total, 'b-', linewidth=2,
                             label=f"{result['label']}")
                axes[i].set_xlabel('t')
                axes[i].set_ylabel('PDF')
                # axes[i].set_ylim(-50, 0)
                axes[i].legend()

            except FileNotFoundError:
                print(f"File {filename} not found!")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        plt.tight_layout()

        imgname = f"pdfs_exp{self.experiment_id}_N{N}"
        img_path = os.path.join(images_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def compare_multiple_pdf(self, results, N):
        if not results:
            print("No data to display")
            return

        fig, ax = plt.subplots()

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink']
        linestyles = ['-', '--', '-.', ':', '--', ':']
        ax.set_prop_cycle(cycler(color=colors) + cycler(linestyle=linestyles))

        for result in results:
            filename = f"{result['name']}_exp{self.experiment_id}_N{N}_out.npz"
            path = os.path.join(data_folder, filename)
            try:
                with np.load(path) as data:
                    dt = data['dt']
                    total = data['ptotal']
                    if len(total.shape) > 1:
                        total = total.flatten()
                    success = data['pTimeN']
                    failure = data['pTime0']

                ax.plot(dt, total, linewidth=2,
                        label=f"{result['label']}")
                ax.set_xlabel('t')
                ax.set_ylabel('PDF')
                ax.set_ylim(-15, -4)
                ax.legend()

            except FileNotFoundError:
                print(f"File {filename} not found!")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        plt.tight_layout()
        imgname = f"compare_pdfs_exp{self.experiment_id}_N{N}"
        img_path = os.path.join(images_folder, imgname)
        plt.savefig(img_path)
        plt.close()


class MultipleOptimizer:
    def __init__(self, solver, profiles, N_ref=100, log_scale=False, experiment_id=0):
        self.N_ref = N_ref
        self.reference_models = {}
        self.solver = solver
        self.multiple = CompareProfiles(profiles=profiles)
        self.profiles = profiles
        self.log_scale = log_scale
        self.experiment_id = experiment_id

    def compute_reference_models(self):
        print(
            f"Computing reference models for N_ref={self.N_ref}, Experiment {self.experiment_id}")

        results = []
        for profile in self.profiles:
            print(f"Computing reference for {profile['label']}")
            result = self.solver.run_fp(
                self.N_ref, profile['type'], profile['params'],
                name=f"{profile['name']}_exp{self.experiment_id}", 
                log_scale=self.log_scale, exp_id=self.experiment_id)
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['params'] = profile["params"]
                result['profile'] = profile
                self.reference_models[profile['name']] = result
                results.append(result)
                print(
                    f"Reference model for {profile['label']} computed successfully")
            else:
                print(
                    f"Failed to compute reference model for {profile['label']}")
                return False
        return True

    def loss_function(self, theta, profile_name):
        theta = theta.flatten()
        if not self.reference_models:
            print("ERROR: Target not computed!")

        N_value = self.N_ref
        profile = next(
            (p for p in self.profiles if p['name'] == profile_name), None)

        # Create new params dictionary based on profile type
        params = profile['params'].copy()

        if profile['type'] == "linear":
            if len(theta) > 0:
                params['slope'] = theta[0]

        elif profile['type'] == "quadratic":
            if len(theta) > 0:
                params['a'] = theta[0]

        elif profile['type'] == "gauss":
            if len(theta) > 0:
                params['A'] = theta[0]

        # Run Fortran program for this params
        result = self.solver.run_fp(
            N_value, profile['type'], params,
            name=f"{profile['name']}_exp{self.experiment_id}_N{N_value}_opt", 
            log_scale=self.log_scale, exp_id=self.experiment_id)

        current_value = result['ptotal']
        target = self.reference_models[profile_name]['ptotal']

        difference = np.mean((current_value - target)**2)  # mse

        return np.array([[difference]])

    def run_multiple_opt(self, initial_points=50,  plot_results=True):

        if not self.reference_models:
            success = self.compute_reference_models()
            if not success:
                print("ERROR: Cannot compute reference model")
                return None

        results = {}

        for profile in self.profiles:
            profile_name = profile['name']

            # Define optimization space
            if profile["type"] == "linear":
                space = [
                    {
                        "name": "a",
                        "type": "continuous",
                        "domain": (-0.5, 0.5),
                        "dimensionality": 1,
                    }
                ]
            elif profile["type"] == "quadratic":
                space = [
                    {
                        "name": "a",
                        "type": "continuous",
                        "domain": (-0.02, 0.02),
                        "dimensionality": 1,
                    },

                ]
            elif profile['type'] == "small_min":
                space = [
                    {
                        "name": "a",
                        "type": "continuous",
                        "domain": (0.001, 0.1),
                        "dimensionality": 1,
                    },
                ]
            elif profile['type'] == "gauss":
                space = [
                    {
                        "name": "A",
                        "type": "continuous",
                        "domain": (0, 8),
                        "dimensionality": 1,

                    }

                ]

            print(
                f"\nStarting optimization for {profile['label']} (Experiment {self.experiment_id})")

            # Select randomly 50 points to use as the initial training set
            feasible_region = GPyOpt.Design_space(space=space)
            initial_design = GPyOpt.experiment_design.initial_design(
                "random", feasible_region, initial_points)

            # Setup optimizer components
            objective = GPyOpt.core.task.SingleObjective(
                lambda theta: self.loss_function(theta, profile_name))

            model = GPyOpt.models.GPModel(
                exact_feval=True,
                optimize_restarts=10,
                verbose=False
            )

            acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(
                feasible_region)
            acquisition = GPyOpt.acquisitions.AcquisitionEI(
                model, feasible_region, optimizer=acquisition_optimizer
            )

            # Choose a sequental collection method
            evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

            # Create and run Bayesian optimizer
            bo = GPyOpt.methods.ModularBayesianOptimization(
                model,
                feasible_region,
                objective,
                acquisition,
                evaluator,
                initial_design,
                normalize_Y=False,
            )

            max_time = None
            max_iter = 10
            tolerance = 1e-8  # Distance between two consecutive observations
            best_params = None

            try:
                for i in range(10):
                    bo.run_optimization(
                        max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

                    best_params = bo.x_opt
                    print(best_params)
                    opt_loss = bo.fx_opt

                    optimized_params = profile['params'].copy()

                    if profile["type"] == "linear":
                        if len(best_params) > 0:
                            optimized_params['slope'] = best_params[0]

                    elif profile["type"] == "quadratic":
                        if len(best_params) > 0:
                            optimized_params['a'] = best_params[0]

                    elif profile['type'] == "gauss":
                        if len(best_params) > 0:
                            optimized_params['A'] = best_params[0]

                    # Print the current best result
                    print(
                        f"Experiment: {self.experiment_id}, Iteration: {(i + 1) * 5}")
                    print(f"Objective function value: {opt_loss}")
                    print(f"Parameters: {optimized_params}")

                    if opt_loss == 0:
                        print(
                            f"Optimization stopped: Best params: {best_params}")
                        break

                    best_result = self.solver.run_fp(
                        self.N_ref, profile['type'], optimized_params,
                        name=f"{profile_name}_exp{self.experiment_id}_best", 
                        log_scale=self.log_scale, exp_id=self.experiment_id)

                    # input("Press Enter to continue...")

            except Exception as e:
                print(f"Error in optimization iteration {i+1}: {e}")
                traceback.print_exc()
                return None

            optimized_params = profile['params'].copy()
            if profile['type'] == "linear":
                if len(best_params) > 0:
                    optimized_params['slope'] = best_params[0]

            elif profile['type'] == "quadratic":
                if len(best_params) > 0:
                    optimized_params['a'] = best_params[0]

            elif profile['type'] == "gauss":
                if len(best_params) > 0:
                    optimized_params['A'] = best_params[0]

            best_result = self.solver.run_fp(
                self.N_ref, profile['type'], optimized_params,
                name=f"{profile_name}_exp{self.experiment_id}_final", 
                log_scale=self.log_scale, exp_id=self.experiment_id)

            if plot_results and best_result:
                ref_model = self.reference_models[profile_name]

                fig, ax = plt.subplots()
                ax.plot(ref_model['dt'], ref_model['ptotal'],
                        'b-', label=f'Reference (N={self.N_ref})')
                ax.plot(best_result['dt'], best_result['ptotal'],
                        'r--', label=f'Optimized')
                ax.set_xlabel('t')
                ax.set_ylabel('PDF')
                ax.set_title(
                    f'{profile["label"]} (Experiment {self.experiment_id})')
                ax.legend()
                plt.tight_layout()

                imgname = f"current_opt_{profile['name']}_exp{self.experiment_id}"
                img_path = os.path.join(images_folder, imgname)
                plt.savefig(img_path)
                plt.close()

            results[profile_name] = {
                'best_params': optimized_params,
                'best_loss': opt_loss,
                'bo_object': bo,
                'final_result': best_result
            }

            print(
                f"Optimization completed for {profile['label']} (Experiment {self.experiment_id}):")
            print(f"Optimized params = {optimized_params}, Loss = {opt_loss}")

        # Compare all optimized results
        if results and plot_results:

            fig, axes = plt.subplots(nrows=3, ncols=1)
            axes = axes.flatten()
            fig.suptitle(
                f"Different optimized profiles (Experiment {self.experiment_id}):")

            for i, profile in enumerate(self.profiles):
                profile_name = profile['name']
                if profile_name in results and results[profile_name]['final_result']:
                    result = results[profile_name]
                    ref_model = self.reference_models[profile_name]
                    axes[i].plot(result['final_result']['dt'], result['final_result']['ptotal'],
                                 label=f"{profile['label']}")
                    axes[i].plot(ref_model['dt'], ref_model['ptotal'],
                                 'r-', label=f'Reference (N={self.N_ref})', linestyle="--")
                    axes[i].set_xlabel('t')
                    axes[i].set_ylabel('PDF')
                    axes[i].legend()

            # plt.tight_layout()
            path = os.path.join(
                images_folder, f"opt_result_exp{self.experiment_id}_N{self.N_ref}")
            plt.savefig(path)
            plt.close()

        return results


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

        self.create_summary_report(results)

        return results

    def create_summary_report(self, results):
        """Create a summary report of all experiments"""
        # Create lists to store the data
        data = []

        for exp_id in range(len(self.profile_sets)):
            exp_key = f'experiment_{exp_id}'
            if exp_key in results:
                for N in self.N_values:
                    if N in results[exp_key]:
                        if 'optimization_results' in results[exp_key][N]:
                            opt_results = results[exp_key][N]['optimization_results']

                            for profile_name, result in opt_results.items():
                                # Add each result as a row in our data
                                row = {
                                    'N': N,
                                    'Profile': profile_name,
                                    'Loss function': result['best_loss'],
                                }

                                # Add best parameters as separate columns
                                for param_name, param_value in result['best_params'].items():
                                    row[f'Param_{param_name}'] = param_value

                                data.append(row)

        # Create DataFrame from the collected data
        df = pd.DataFrame(data)

        filename = f"{exp_id}.csv"
        # Save to CSV file
        output_path = os.path.join(data_folder, filename)
        df.to_csv(output_path, index=False)


# Example usage
if __name__ == "__main__":  # Preventing unwanted code execution during import

    parser = argparse.ArgumentParser(
        description="Run the experiment with profiles in .sh file")
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
        default=True,
        help="Use logarithmic scale"
    )
    args = parser.parse_args()

    # Initialize solver
    solver = FokkerPlanckSolver()

    if args.profiles_json:
        # Parse the JSON string containing all profile sets
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
    else:
        # Define different profile types and parameters
        profiles_1 = [
            {"type": "linear", "params": {"slope": -0.1},
                "label": "linear (slope=-0.1)", "name": "pr1"},
            {"type": "linear", "params": {"slope": -0.07},
             "label": "linear (slope=-0.07)", "name": "pr2"},
            {"type": "linear", "params": {"slope": -0.08},
             "label": "linear (slope=-0.08)", "name": "pr3"},]

        profiles_2 = [
            {"type": "quadratic", "params": {"a": 0.007, "b": 25, "c": -4},
                "label": "Quadratic (a=0.007)", "name": "pr4"},
            {"type": "quadratic", "params": {"a": 0.008, "b": 25, "c": -5},
                "label": "Quadratic (a=0.008)", "name": "pr5"},
            {"type": "quadratic", "params": {"a": 0.01, "b": 25, "c": -6},
             "label": "Quadratic (a=0.01)", "name": "pr6"},
        ]

        profiles_3 = [
            {"type": "quadratic", "params": {"a": 0.012, "b": 25, "c": -7},
                "label": "Quadratic (a=0.012)", "name": "pr7"},
            {"type": "quadratic", "params": {"a": 0.008, "b": 25, "c": -5},
                "label": "Quadratic (a=0.008)", "name": "pr5"},
            {"type": "quadratic", "params": {"a": 0.01, "b": 25, "c": -6},
             "label": "Quadratic (a=0.01)", "name": "pr6"},
        ]

        profiles_4 = [
            {"type": "gauss", "params": {"A": 1},
                "label": "Gauss (A=1)", "name": "pr8"},
            {"type": "gauss", "params": {"A": 3},
                "label": "Gauss (A=3)", "name": "pr9"},
            {"type": "gauss", "params": {"A": 0},
             "label": "Gauss (A=0)", "name": "pr10"},
        ]

        profiles_5 = [{"type": "gauss", "params": {"A": 1},
                       "label": "Gaussian (A=1)", "name": "pr8"},
                      {"type": "gauss", "params": {"A": 3},
                       "label": "Gaussian (A=3)", "name": "pr9"},
                      {"type": "gauss", "params": {"A": 8},
                       "label": "Gaussain (A=8)", "name": "pr11"},
                      ]

        # Create an experiment series
        experiment_series = ExperimentSeries(
            solver=solver,
            profile_sets=[profiles_1, profiles_2,
                          profiles_3, profiles_4, profiles_5],
            N_values=[50, 100],
            log_scale=True
        )

        # Run experiments
        results = experiment_series.run_experiments()

        print("All experiments completed successfully!")

    # Clean up data files
    npz_files = glob.glob(os.path.join(data_folder, '*.npz'))
    for file in npz_files:
        os.remove(file)
