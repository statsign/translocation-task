from main import FokkerPlanckSolver, BayesOptimizer
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import GPyOpt
import traceback
import os
import glob

png_files = glob.glob(os.path.join("images", '*.png'))

for file in png_files:
    os.remove(file)


class CompareProfiles:
    def __init__(self, profiles=None):

        # Initialize solver
        self.solver = FokkerPlanckSolver()

        if profiles == None:
            # Define different profile types and parameters
            self.profiles = [
                {"type": "linear", "params": {"slope": -0.01},
                 "label": "Linear (slope=-0.01)", "name": "pr1"},
                {"type": "linear", "params": {"slope": -0.02},
                 "label": "Linear (slope=-0.02)", "name": "pr2"},
                {"type": "small_min", "params": {"a": 60, "t": 90},
                 "label": "Small minimum (a=60)", "name": "pr3"},
                {"type": "small_min", "params": {"a": 40, "t": 90},
                 "label": "Small minimum (a=40)", "name": "pr4"}
            ]

        else:
            self.profiles = profiles

    def run_multiple_simulations(self, N):
        self.plot_profiles(N)

        results = []
        for profile in self.profiles:
            result = self.solver.run_fp(
                N, profile_type=profile['type'], params=profile['params'], name=profile['name'])
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['profile'] = profile
                results.append(result)

        self.plot_multiple_pdf(results, N)

        return results

    def plot_profiles(self, N):

        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        for i, profile in enumerate(self.profiles):
            # Generate the profile first
            self.solver.gen_profile(
                N, profile['type'], profile['params'], profile['name'])
            filename = profile["name"] + f"_{N}" + "_in" + '.npz'
            with np.load(filename) as data:
                zn = data['dt']
                F = data['F']
            axes[i].plot(zn, F, label=f"{profile['label']}")
            axes[i].legend(loc='best')
        filename = f"profiles_{N}"
        path = os.path.join("images", filename)
        plt.savefig(path)
        plt.close()

    def plot_multiple_pdf(self, results, N):
        if not results:
            print("No data to display")
            return

        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        for i, result in enumerate(results):
            filename = result["name"] + f"_{N}" + '_out' + '.npz'
            file_path = os.path.join(".", filename)
            try:
                os.path.exists(file_path)
                with np.load(filename) as data:

                    dt = data['dt']
                    total = data['ptotal']
                    success = data['pTimeN']
                    failure = data['pTime0']

                axes[i].plot(dt, total, 'b-', linewidth=2,
                             label=f"{result['label']}")
                axes[i].set_xlabel('t')
                axes[i].set_ylabel('PDF')
                axes[i].legend()

            except FileNotFoundError:
                print(f"File {filename} not found!")

        filename = f"pdfs_{results[0]['N']}"
        path = os.path.join("images", filename)
        plt.savefig(path)
        plt.close()


class MultipleOptimizer:
    def __init__(self, solver, N_ref=100, profiles=None):
        self.N_ref = N_ref
        self.reference_models = {}
        self.solver = solver
        self.multiple = CompareProfiles(profiles=profiles)
        self.profiles = profiles

    def compute_reference_models(self):
        print(f"Computing reference models for N_ref={self.N_ref}")

        results = []
        for profile in self.profiles:
            print(f"Computing reference for {profile['label']}")
            result = self.solver.run_fp(
                self.N_ref, profile['type'], profile['params'])
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['profile'] = profile
                self.reference_models[profile['name']] = result
                results.append(result)
                print(
                    f"Reference model for {profile['label']} computed successfully")
            else:
                print(
                    f"Failed to compute reference model for {profile['label']}")
                return False

    def loss_function(self, theta, profile_name):
        if self.reference_models is None:
            print("ERROR: Target not computed!")

        N_value = int(theta[0])
        profile = next(
            (p for p in self.profiles if p['name'] == profile_name), None)

        # Run Fortran program for this N value
        result = self.solver.run_fp(
            N_value, profile['type'], profile['params'])

        current_value = result['ptotal']
        target = self.reference_models[profile_name]['ptotal']

        difference = np.mean((current_value - target)**2)  # mse

        # Write to log file
        with open("optimization_log.txt", "a") as f:
            f.write(
                f"Profile={profile['label']}, N={N_value}, Loss={difference}\n")

        return np.array([[difference]])

    def run_multiple_opt(self, max_iter=5, initial_points=50, domain=(2, 500), plot_results=True):

        if self.reference_models is None:
            success = self.compute_reference_models()
            if not success:
                print("ERROR: Cannot compute reference model")
                return None

        # Define optimization space
        space = [
            {
                "name": "var_1",
                "type": "discrete",
                "domain": tuple(range(domain[0], domain[1] + 1)),
                "dimensionality": 1,
            }
        ]

        results = {}

        for profile in self.profiles:
            profile_name = profile['name']
            print(f"\nStarting optimization for {profile['label']}")

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
            max_iter = 5
            tolerance = 1e-25  # Distance between two consecutive observations

            try:
                for i in range(10):
                    bo.run_optimization(
                        max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

                    best_N = int(bo.x_opt[0])
                    opt_loss = bo.fx_opt

                    # Print the current best result
                    print(f"Iteration: {(i + 1) * 5}")
                    print(f"Best N value: {best_N}")
                    print(f"Objective function value: {opt_loss}")

                    # Check if the best N is the reference N
                    if int(best_N) == self.N_ref:
                        print(
                            f"Optimization stopped: Best N={best_N} equals N_ref={self.N_ref}")
                        best_result = self.reference_models[profile_name]
                        break

                    best_result = self.solver.run_fp(
                        best_N, profile['type'], profile['params'])

                    # input("Press Enter to continue...")

            except Exception as e:
                print(f"Error in optimization iteration {i+1}: {e}")
                traceback.print_exc()
                return None

            best_result = self.solver.run_fp(
                best_N, profile['type'], profile['params'])

            if plot_results and best_result:
                ref_model = self.reference_models[profile_name]

                fig, ax = plt.subplots()
                ax.plot(ref_model['dt'], ref_model['ptotal'],
                        'b-', label=f'Reference (N={self.N_ref})')
                ax.plot(best_result['dt'], best_result['ptotal'],
                        'r--', label=f'Optimized (N={best_N})')
                ax.set_xlabel('t')
                ax.set_ylabel('PDF')
                ax.set_title(f'{profile["label"]}')
                ax.legend()
                plt.tight_layout()

                filename = f"current_opt_{profile['name']}_{best_N}"
                path = os.path.join("images", filename)
                plt.savefig(path)
                plt.close()

            results[profile_name] = {
                'best_N': best_N,
                'best_loss': opt_loss,
                'bo_object': bo,
                'final_result': best_result
            }

            print(
                f"Optimization completed for {profile['label']}: Best N = {best_N}, Loss = {opt_loss}")

        # Compare all optimized results
        if results and plot_results:
            ref_model = self.reference_models[profile_name]
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes = axes.flatten()
            fig.suptitle('Different optimized profiles')

            for i, profile in enumerate(self.profiles):
                profile_name = profile['name']
                if profile_name in results and results[profile_name]['final_result']:
                    result = results[profile_name]
                    axes[i].plot(result['final_result']['dt'], result['final_result']['ptotal'],
                                 label=f"{profile['label']} (N={result['best_N']})")
                    axes[i].plot(ref_model['dt'], ref_model['ptotal'],
                                 'r-', label=f'Reference (N={self.N_ref})', linestyle="--")
                    axes[i].set_xlabel('t')
                    axes[i].set_ylabel('PDF')
                    axes[i].legend()

            plt.tight_layout()
            path = os.path.join("images", "opt_result")
            plt.savefig(path)
            plt.close()

        return results


# Define different profile types and parameters
profiles = [
    {"type": "linear", "params": {"slope": -0.01},
        "label": "Linear (slope=-0.01)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.02},
        "label": "Linear (slope=-0.02)", "name": "pr2"},
    {"type": "small_min", "params": {"a": 60, "t": 90, "c": -0.02},
        "label": "Small minimum (a=60)", "name": "pr3"},
    {"type": "small_min", "params": {"a": 40, "t": 90, "c": -0.02},
        "label": "Small minimum (a=40)", "name": "pr4"}
]

# Example usage
if __name__ == "__main__":  # Preventing unwanted code execution during import
    # Initialize solver
    solver = FokkerPlanckSolver()
    compare = CompareProfiles(profiles=profiles)

    # Initialize optimizer
    N_0 = 100
    optimizer = MultipleOptimizer(solver, N_ref=N_0, profiles=profiles)

    # Compute reference models
    optimizer.compute_reference_models()

    # Plot reference model profile and distribution
    compare.run_multiple_simulations(N_0)

    # Run optimization
    results = optimizer.run_multiple_opt()


npz_files = glob.glob('*.npz')

for file in npz_files:
    os.remove(file)
