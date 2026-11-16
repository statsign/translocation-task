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
import csv


data_folder, images_folder = get_output_folders()

# lower bound: above the rounding noise of the solver output
EPS_FLOOR = {"log_mse_split": 1e-10, "mse": 1e-16}
LOG_FLOOR = 1e-40  # same clamp for pTimeN and pTime0; the Fortran output clamps them differently

def plot_convergence(histories, labels, path, title=None, log_scale=True):
    """
    Convergence of the algorithm

    Args:
        histories: list of dicts from MultipleOptimizer.convergence_history
        labels: legend label for each history
        log_scale: log y axis; values below eps are drawn at eps
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for h, label in zip(histories, labels):
        step = np.asarray(h['step'])
        best = np.asarray(h['best_loss'], dtype=float)
        if log_scale:
            best = np.maximum(best, float(h['eps']))  # below eps differences are not meaningful
        line, = ax.plot(step, best, drawstyle='steps-post', label=label)
        ax.scatter(step, best, marker='+', color=line.get_color())
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('bayesian optimization steps')
    ax.set_ylabel('best f')
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

class MultipleOptimizer:
    def __init__(self, solver, profiles, N_ref=100, experiment_id=1,
                 log_loss=True, eps=None, eps_rel=1e-6, eps_floor=None,
                 loss="log_mse_split"):
        """
        Args:
            solver: FokkerPlanckSolver instance
            N_ref: Reference N value for comparison
            reference_model: A dictionary of reference results
            loss: discrepancy between candidate and reference
                "log_mse_split": MSE(ln[success_rate * pTimeN]) + MSE(ln[failure_rate * pTime0]),
                    success and failure compared separately, weighted by their rates
                "mse": MSE over ptotal
        """
        if loss not in EPS_FLOOR:
            raise ValueError(f"Unknown loss: {loss}. Available: {list(EPS_FLOOR)}")
        self.loss = loss
        if eps_floor is None:
            eps_floor = EPS_FLOOR[loss]

        self.N_ref = N_ref
        self.reference_models = {}
        self.solver = solver
        self.multiple = CompareProfiles(profiles=profiles)
        self.profiles = profiles
        self.experiment_id = experiment_id

        # Log-transform of the loss: y = log10(loss + eps)
        self.log_loss = log_loss
        self.eps = eps # if None, chosen from the initial design
        self.eps_rel = eps_rel # eps = eps_rel * median(loss over initial points)
        self.eps_floor = eps_floor # lower bound: rounding noise of the Fortran output (e13.6)
        self.eps_by_profile = {} # eps per profile: loss scale differs between profiles
        self.worst_y = {} # worst y seen so far, per profile; used as a penalty for failed solver runs
        self.histories = {} # convergence history per profile, see convergence_history

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
                exp_id=self.experiment_id)
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

    def raw_loss(self, theta, profile_name):
        """
        Discrepancy between the candidate and the reference model

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
            name=profile['name'], exp_id=self.experiment_id,
            save=False)

        reference = self.reference_models[profile_name]

        if self.loss == "mse":
            return float(np.mean((result['ptotal'] - reference['ptotal'])**2))

        if self.loss == "log_mse_split":
            # p(outcome, t) = rate * p(t | outcome): without the rates, profiles with
            # the same time shapes but different success rates (e.g. slope k and -k) look identical
            weights = {'pTimeN': 'success_rate', 'pTime0': 'failure_rate'}
            return float(sum(
                np.mean((np.log(np.maximum(result[key] * result[w], LOG_FLOOR))
                        - np.log(np.maximum(reference[key] * reference[w], LOG_FLOOR)))**2)
                for key, w in weights.items()))

    def transform(self, raw_loss, profile_name):
        """loss -> value seen by the GP"""
        if not self.log_loss:
           return raw_loss
        return np.log10(raw_loss + self.eps_by_profile[profile_name])

    def inverse_transform(self, y, profile_name):
        """GP value -> loss (for logging and the stopping criterion)"""
        if not self.log_loss:
            return y
        return 10.0**y - self.eps_by_profile[profile_name]

    def choose_eps(self, loss_init):
        if self.eps is not None:
            return self.eps
        finite = loss_init[np.isfinite(loss_init)]
        return max(self.eps_rel * np.median(finite), self.eps_floor)

    def loss_function(self, theta, profile_name):
        mse = self.raw_loss(theta, profile_name)
        y = self.transform(mse, profile_name)
        if not np.isfinite(y): # solver failed: penalty = worst value seen so far      
            y = self.worst_y.get(profile_name, 0.0)
        self.worst_y[profile_name] = max(self.worst_y.get(profile_name, y), y)
        return np.array([[y]])

    def convergence_history(self, bo, profile, n_init):
        """
        Best loss found so far at each BO step.
        Step 0 is the best point of the initial design (first n_init evaluations).
        """
        name = profile['name']
        loss = np.array([self.inverse_transform(y, name) for y in bo.Y.ravel()])
        # now it is not the exact objective, will be fixed soon
        best = np.minimum.accumulate(loss)[n_init - 1:]
        return {
            'loss_name': self.loss,
            'step': np.arange(len(best)),
            'best_loss': best,
            'eps': self.eps_by_profile[name],
        }

    def params_table(self, results):
        """
        Table: reference vs recovered parameters
        """
        params, rows = [], []
        for profile in self.profiles:
            name = profile['name']
            if name not in results:
                continue
            defaults = getattr(self.solver, 'PROFILE_DEFAULTS', {}).get(profile['type'], {})
            reference = {**defaults, **profile['params']}  # missing parameters = solver defaults
            recovered = results[name]['best_params']
            # optimized parameters = names in the optimization space of this profile
            optimized = [v['name'] for v in results[name]['bo_object'].space.config_space]
            row = {'profile': profile['label']}
            for p in optimized:
                if p not in params:
                    params.append(p)
                row[f'{p}_ref'] = float(reference[p])
                row[f'{p}_rec'] = float(recovered[p])
            rows.append(row)

        columns = ['profile'] + [f'{p}_{kind}' for p in params for kind in ('ref', 'rec')]

        folder = os.path.join(data_folder, "opt", f"exp{self.experiment_id}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"params_exp{self.experiment_id}_N{self.N_ref}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval='')
            writer.writeheader()
            writer.writerows(rows)

        return rows

    def plot_profiles(self, results):
        """
        Recovered free energy profiles F(s) 
        """
        names = [p['name'] for p in self.profiles if p['name'] in results]
        ncols = min(len(names), 3)
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        s = np.arange(self.N_ref + 1)

        for ax, name in zip(axes.flat, names):
            profile = next(p for p in self.profiles if p['name'] == name)
            F_ref = self.solver.profile_func(s, profile['type'], profile['params'])
            F_rec = self.solver.profile_func(s, profile['type'], results[name]['best_params'])
            ax.plot(s, F_ref, color='b', linestyle='-', label='Reference')
            ax.plot(s, F_rec, color='r', linestyle='--', label='Recovered')
            ax.set_title(profile['label'])
            ax.set_xlabel('s')
            ax.set_ylabel('F(s)/kT')
            ax.set_xlim(0, self.N_ref)
            ax.legend()
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)  # unused panels

        fig.suptitle(f"Recovered profiles (Experiment {self.experiment_id}, N={self.N_ref})")
        plt.tight_layout()
        img_folder = os.path.join(images_folder, "opt", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        plt.savefig(os.path.join(img_folder, f"profiles_exp{self.experiment_id}_N{self.N_ref}"))
        plt.close(fig)
        
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
                        "domain": (4, 6),
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


            # evaluate the loss on the initial design ourselves to choose eps

            loss_init = np.array([self.raw_loss(x, profile_name) for x in initial_design])
            eps = self.choose_eps(loss_init)
            self.eps_by_profile[profile_name] = eps
            Y_init = np.array([self.transform(m, profile_name) for m in loss_init])
            bad = ~np.isfinite(Y_init)
            Y_init[bad] = np.nanmax(Y_init[~bad])
            self.worst_y[profile_name] = Y_init.max()
            Y_init = Y_init.reshape(-1, 1)
            logger.info(f"eps = {eps:.3e}, {self.loss} init: min={np.nanmin(loss_init):.3e}, "
                        f"median={np.nanmedian(loss_init):.3e}")

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
                Y_init=Y_init,
                normalize_Y=True,
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
                        f"Experiment: {self.experiment_id}, rounds: {i + 1}, "
                        f"evaluations: {len(bo.Y)}")
                    best_raw_loss = self.inverse_transform(opt_loss, profile_name)
                    logger.info(f"Objective: y={opt_loss:.4f}, {self.loss}={best_raw_loss:.3e}")
                    logger.info(f"Parameters: {optimized_params}")

                    if best_raw_loss <= eps: # below the level where differences are meaningful
                        logger.info(
                            f"Optimization stopped: Best params: {best_params}")
                        break

                    # input("Press Enter to continue...")

            except Exception as e:
                logger.error(f"Error in optimization iteration {i+1}: {e}")
                traceback.print_exc()
                return None

            history = self.convergence_history(bo, profile, n_init=len(initial_design))
            self.histories[profile_name] = history
            hist_name = f"convergence_{profile_name}_exp{self.experiment_id}_N{self.N_ref}"
            hist_folder = os.path.join(data_folder, "opt", f"exp{self.experiment_id}")
            os.makedirs(hist_folder, exist_ok=True)
            np.savez(os.path.join(hist_folder, hist_name + ".npz"), **history)

            if plot_results:
                img_folder = os.path.join(images_folder, "opt", f"exp{self.experiment_id}")
                os.makedirs(img_folder, exist_ok=True)
                plot_convergence([history], [self.loss], os.path.join(img_folder, hist_name),
                                 title=f'{profile["label"]} (Experiment {self.experiment_id})')

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
                exp_id=self.experiment_id)

            if plot_results and best_result:
                ref_model = self.reference_models[profile_name]

                panels = [('ptotal', 'p(t) (total)'),
                          ('pTimeN', 'p_T(t) (success)'),
                          ('pTime0', 'p_F(t) (failure)')]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, (key, title) in zip(axes, panels):
                    plot_pdf(ax, ref_model['dt'], ref_model[key], 
                             color='b', linestyle='-', label='Reference')
                    plot_pdf(ax, best_result['dt'], best_result[key], 
                             color='r', linestyle='--', label='Optimized')
                    ax.set_title(title)
                    ax.set_xlabel('t')
                    ax.set_ylabel('p(t)')
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
                'best_loss': self.inverse_transform(opt_loss, profile_name),  # loss
                'loss_name': self.loss,
                'best_y': opt_loss,    
                'bo_object': bo,
                'history': history,
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
                    plot_pdf(ax, ref['dt'], ref[key], 
                             color='b', linestyle='-', label='Reference')
                    plot_pdf(ax, opt['dt'], opt[key],
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

        if results:
            self.params_table(results)
            if plot_results:
                self.plot_profiles(results)

        return results