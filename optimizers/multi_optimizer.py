from experiments.compare_profiles import CompareProfiles
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt
import traceback
import os
from utils.path_utils import get_output_folders

data_folder, images_folder = get_output_folders()


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
        """
        Goes through all the profiles and runs Fokker Planck, then saves the results as references
        """
        print(
            f"Computing reference models for N_ref={self.N_ref}, Experiment {self.experiment_id}")

        results = []
        for profile in self.profiles:
            print(f"Computing reference for {profile['label']}")
            result = self.solver.run_fp(
                self.N_ref, profile['type'], profile['params'],
                name=profile['name'],
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
            if len(theta) >= 1:
                params['slope'] = theta[0]

        elif profile['type'] == "gauss":
            param_idx = 0
            if param_idx < len(theta):
                params['A'] = theta[param_idx]
                param_idx += 1
            if param_idx < len(theta):
                params['sigma'] = theta[param_idx]
                param_idx += 1
            if param_idx < len(theta):
                params['k'] = theta[param_idx]
                param_idx += 1

        # Run Fortran program for this params
        result = self.solver.run_fp(
            N_value, profile['type'], params,
            name=profile['name'], log_scale=self.log_scale, exp_id=self.experiment_id)

        current_value = result['ptotal']
        target = self.reference_models[profile_name]['ptotal']

        difference = np.mean((current_value - target)**2)  # mse

        return np.array([[difference]])

    def run_multiple_opt(self, max_iter=10, initial_points=50,  plot_results=True):

        if not self.reference_models:
            success = self.compute_reference_models()
            if not success:
                print("ERROR: Cannot compute reference model")
                return None

        results = {}

        for profile in self.profiles:
            profile_name = profile['name']

            # Define optimization space based on profile type

            if profile["type"] == "linear":
                space = [
                    {
                        "name": "slope",
                        "type": "continuous",
                        "domain": (-0.5, 0.5),
                        "dimensionality": 1,
                    }
                ]

            elif profile['type'] == "gauss":
                space = [
                    {
                        "name": "A",
                        "type": "continuous",
                        "domain": (-4, 4),
                        "dimensionality": 1,

                    },
                    {
                        "name": "sigma",
                        "type": "continuous",
                        "domain": (0.1, 10),
                        "dimensionality": 1,
                    },
                    {"name": "k",
                     "type": "continuous",
                     "domain": (-0.5, 0.5),
                     "dimensionality": 1
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
            tolerance = 1e-8  # Distance between two consecutive observations

            try:
                for i in range(10):
                    bo.run_optimization(
                        max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

                    best_params = bo.x_opt
                    opt_loss = bo.fx_opt

                    optimized_params = profile['params'].copy()

                    if profile['type'] == "linear":
                        if len(best_params) >= 1:
                            optimized_params['slope'] = best_params[0]

                    elif profile['type'] == "gauss":
                        param_idx = 0
                        if param_idx < len(best_params):
                            optimized_params['A'] = best_params[param_idx]
                            param_idx += 1

                        if param_idx < len(best_params):
                            optimized_params['sigma'] = best_params[param_idx]
                            param_idx += 1

                        if param_idx < len(best_params):
                            optimized_params['k'] = best_params[param_idx]
                            param_idx += 1

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
                        name=profile_name,
                        log_scale=self.log_scale, exp_id=self.experiment_id)

                    # input("Press Enter to continue...")

            except Exception as e:
                print(f"Error in optimization iteration {i+1}: {e}")
                traceback.print_exc()
                return None

            optimized_params = profile['params'].copy()
            if profile['type'] == "linear":
                if len(best_params) >= 1:
                    optimized_params['slope'] = best_params[0]

            elif profile['type'] == "gauss":
                param_idx = 0
                if param_idx < len(best_params):
                    optimized_params['A'] = best_params[param_idx]
                    param_idx += 1

                if param_idx < len(best_params):
                    optimized_params['sigma'] = best_params[param_idx]
                    param_idx += 1

                if param_idx < len(best_params):
                    optimized_params['k'] = best_params[param_idx]
                    param_idx += 1

            best_result = self.solver.run_fp(
                self.N_ref, profile['type'], optimized_params,
                name=profile_name,
                log_scale=self.log_scale, exp_id=self.experiment_id)

            if plot_results and best_result:
                ref_model = self.reference_models[profile_name]

                fig, ax = plt.subplots()
                ax.plot(ref_model['dt'], ref_model['ptotal'],
                        'b-', label=f'Reference')
                ax.plot(best_result['dt'], best_result['ptotal'],
                        'r--', label=f'Optimized')
                ax.set_xlabel('t')
                if self.log_scale == False:
                    ax.set_ylabel('p(t)')
                else:
                    ax.set_ylabel('p(t) log scale')
                    ax.set_yscale('log')
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
                    axes[i].plot(result['final_result']['dt'], result['final_result']['ptotal'], 'g',
                                 label=f"{profile['label']}")
                    axes[i].plot(ref_model['dt'], ref_model['ptotal'],
                                 'r', label=f'Reference', linestyle='--')
                    axes[i].set_xlabel('t')
                    if self.log_scale == False:
                        axes[i].set_ylabel('p(t)')
                    else:
                        axes[i].set_ylabel('p(t) log scale')
                        axes[i].set_yscale('log')
                    axes[i].legend()

            # plt.tight_layout()
            path = os.path.join(
                images_folder, f"opt_result_exp{self.experiment_id}_N{self.N_ref}")
            plt.savefig(path)
            plt.close()

        return results