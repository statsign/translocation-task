from experiments.compare_profiles import CompareProfiles
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt
import traceback
import os
from src.utils.path_utils import get_output_folders
from src.utils.plot_utils import plot_pdf
import logging
logger = logging.getLogger(__name__) 


data_folder, images_folder = get_output_folders()


class MultipleOptimizer:
    def __init__(self, solver, profiles, N_ref=100, log_scale=False, experiment_id=1):
        """
        Args:
            solver: FokkerPlanckSolver instance
            N_ref: Reference N value for comparison
            reference_model: A dictionary of reference results
        """
        self.N_ref = N_ref
        self.reference_models = {}
        self.solver = solver
        self.multiple = CompareProfiles(profiles=profiles)
        self.profiles = profiles
        self.log_scale = log_scale
        self.experiment_id = experiment_id

    def compute_reference_models(self):
        """
        Goes through all profiles and runs Fokker Planck, then saves the results as references
        """
        logger.info(
            f"Computing reference models for N_ref={self.N_ref}, Experiment {self.experiment_id}")

        results = []
        for profile in self.profiles:
            logger.info(f"Computing reference for {profile['label']}")
            result = self.solver.run_fp(
                self.N_ref, profile['type'], profile['params'],
                name=f"{profile['name']}_ref",
                log_scale=self.log_scale, exp_id=self.experiment_id)
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['params'] = profile["params"]
                result['profile'] = profile
                self.reference_models[profile['name']] = result
                results.append(result)
                logger.info(
                    f"Reference model for {profile['label']} computed successfully")
            else:
                logger.error(
                    f"Failed to compute reference model for {profile['label']}")
                return False
        return True

    def loss_function(self, theta, profile_name):
        """
        Loss function for Bayesian optimization

        This function does mse for
            ptotal = pTime0 * failure_rate + pTimeN * success_rate
            pTime0: time distribution for failure
            pTimeN: time distribution for success

        Args:
            theta: array-like parameter

        Returns:
            Loss value based on difference from reference model
        """
        theta = theta.flatten()
        if not self.reference_models:
            logger.error("Target not computed!")

        N_value = self.N_ref
        profile = next((p for p in self.profiles if p['name'] == profile_name), None)

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
            name=profile['name'], log_scale=self.log_scale, exp_id=self.experiment_id,
            save=False)

        current_value = result['ptotal']
        target = self.reference_models[profile_name]['ptotal']

        difference = np.mean((current_value - target)**2)  # mse

        return np.array([[difference]])

    def run_multiple_opt(self, max_iter=10, initial_points=50,  plot_results=True):

        if not self.reference_models:
            success = self.compute_reference_models()
            if not success:
                logger.error("ERROR: Cannot compute reference model")
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

            logger.info(
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
                    logger.info(
                        f"Experiment: {self.experiment_id}, Iteration: {(i + 1) * 5}")
                    logger.info(f"Objective function value: {opt_loss}")
                    logger.info(f"Parameters: {optimized_params}")

                    if opt_loss == 0:
                        logger.info(
                            f"Optimization stopped: Best params: {best_params}")
                        break
                    
                    """
                    best_result = self.solver.run_fp(
                        self.N_ref, profile['type'], optimized_params,
                        name=profile_name,
                        log_scale=self.log_scale, exp_id=self.experiment_id)
                    """

                    # input("Press Enter to continue...")

            except Exception as e:
                logger.error(f"Error in optimization iteration {i+1}: {e}")
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
                name=f"{profile_name}_opt",
                log_scale=self.log_scale, exp_id=self.experiment_id)

            if plot_results and best_result:
                ref_model = self.reference_models[profile_name]

                panels = [('ptotal', 'p(t) (total)'),
                          ('pTimeN', 'p_T(t) (success)'),
                          ('pTime0', 'p_F(t) (failure)')]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, (key, title) in zip(axes, panels):
                    plot_pdf(ax, ref_model['dt'], ref_model[key], self.log_scale,
                             color='b', linestyle='-', label='Reference')
                    plot_pdf(ax, best_result['dt'], best_result[key], self.log_scale,
                             color='r', linestyle='--', label='Optimized')
                    ax.set_title(title)
                    ax.set_xlabel('t')
                    ax.set_ylabel('log scale' if self.log_scale else 'p(t)')
                    ax.legend()
                fig.suptitle(
                    f'{profile["label"]} (Experiment {self.experiment_id}) | '
                    f'success rate: ref={ref_model["success_rate"]:.3g}, '
                    f'opt={best_result["success_rate"]:.3g}')
                plt.tight_layout()

                imgname = f"current_opt_{profile['name']}_exp{self.experiment_id}_N{self.N_ref}"
                img_folder = os.path.join(images_folder, "opt", f"exp{self.experiment_id}")
                os.makedirs(img_folder, exist_ok=True)
                img_path = os.path.join(img_folder, imgname)
                plt.savefig(img_path)
                plt.close()

            results[profile_name] = {
                'best_params': optimized_params,
                'best_loss': opt_loss,
                'bo_object': bo,
                'final_result': best_result
            }

            logger.info(
                f"Optimization completed for {profile['label']} (Experiment {self.experiment_id}):")
            logger.info(f"Optimized params = {optimized_params}, Loss = {opt_loss}")

        if results and plot_results:

            panels = [('ptotal', 'p(t) (total)'),
                        ('pTimeN', 'p_T(t) (success)'),
                        ('pTime0', 'p_F(t) (failure)')]
            n = len(self.profiles)
            fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n), squeeze=False)
            fig.suptitle(f"Optimized profiles (Experiment {self.experiment_id}, N={self.N_ref})")
 
            for row, prof in enumerate(self.profiles):
                name = prof['name']
                if name not in results or not results[name]['final_result']:
                    continue
                opt = results[name]['final_result']  
                ref = self.reference_models[name]       
                for col, (key, title) in enumerate(panels):
                    ax = axes[row, col]
                    plot_pdf(ax, ref['dt'], ref[key], self.log_scale,
                             color='b', linestyle='-', label='Reference')
                    plot_pdf(ax, opt['dt'], opt[key], self.log_scale,
                             color='r', linestyle='--', label='Optimized')
                    ax.set_title(title)
                    ax.set_xlabel('t')
                    ax.legend(fontsize=8)
                axes[row, 0].set_ylabel(
                    f"{prof['label']}\nsuccess: ref={ref['success_rate']:.3g}, "
                    f"opt={opt['success_rate']:.3g}", fontsize=9)
                
            plt.tight_layout()
            imgname = f"opt_result_{profile['name']}_exp{self.experiment_id}_N{self.N_ref}"
            img_folder = os.path.join(images_folder, "opt", f"exp{self.experiment_id}")
            os.makedirs(img_folder, exist_ok=True)
            img_path = os.path.join(img_folder, imgname)
            plt.savefig(img_path)
            plt.close()

        return results