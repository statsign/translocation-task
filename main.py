import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import subprocess
import traceback
import os
import glob

npz_files = glob.glob(os.path.join("data", '*.npz'))

for file in npz_files:
    os.remove(file)


class FokkerPlanckSolver:
    def __init__(self):
        '''

        timesteps (Nt): actual number of timesteps
        dt: timestep (conversion into real units discussed)

        '''

        # Input file parameters
        self.timesteps = 100000
        self.dt = 0.01
        self.timedisplay = 10000
        self.intdisplay = 10

        self.compile_fortran()

    def compile_fortran(self):
        """Compiles the Fortran program"""
        try:
            compile_process = subprocess.run(
                ["gfortran", "fp.f90", "-o", "output"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if compile_process.returncode == 0:
                print("Successfull compilation\n")

            else:
                print("Compilation error:")
                print(compile_process.stderr)

        except Exception as e:
            print(f"Compilation error: {e}")

    def profile_func(self, i, profile_type="linear", params=None):
        """
        Generates free energy profile shape

        Args:
            i: coordinate for each point
            profile_type: type of profile ('linear', 'small_min', ...)

        Returns:
            Profile value at position i
        """

        if params is None:
            params = {}

        if profile_type == "linear":
            slope = params.get('slope', -0.01)
            c = params.get('c', 0)
            return slope * i + c

        elif profile_type == "small_min":
            a = params.get('a', 0.0001)
            t = params.get('t', 0.001)
            c = params.get('c', -0.0002)
            return a * np.exp(-i / t) + c

        else:
            return -i/100  # Default linear profile

    def gen_profile(self, N, profile_type="linear", params=None):
        """
        Generates the input profile for a given N, makes an input file.

        Args:
            N: scalar
                actual number of segments

            profile_type: str
                type of profile ('linear', 'small_min')

            z: scalar
                actual length of the field (related to dt and the mobility constant)

        """
        N = int(N)
        z = N

        dt = []
        F = []

        with open("new_input.txt", "w", encoding="UTF-8") as fin:
            fin.write(f'{N} {z}\n')
            fin.write(
                f"{self.timesteps} {self.dt} {self.timedisplay} {self.intdisplay}\n")

            for i in range(N+1):
                profile_value = self.profile_func(i, profile_type, params)
                fin.write(f"{i} {profile_value:.5e}\n")
                dt.append(i)
                F.append(profile_value)

        result = {
            "N": N,
            "dt": np.array(dt),
            "F": np.array(F),
        }

        return result

    def run_fp(self, N, profile_type="linear", params=None, name="pr", log_scale=False):
        """
        Runs Fokker-Planck program with specified parameters

        Args:
            N: scalar
                actual number of segments
            profile_type: type of profile ('linear', 'small_min', ...)

        Returns:
            Dictionary with simulation results
        """

        # Generate input profile
        self.gen_profile(N, profile_type, params)

        # Run Fortran program
        try:
            subprocess.run(
                ["./output"],
                capture_output=True,
                text=True
            )
            print(f"Calculation for N={N} completed!")

        except subprocess.CalledProcessError as e:
            print(f"Error running program for N={N}: {e}")
            print(f"Stderr: {e.stderr}")
            return None

        # Read results
        result = self.read_pdf(N, name, log_scale)
        if result:
            result['N'] = int(N)
            result['profile_type'] = profile_type
            result['profile_params'] = params

        return result

    def read_pdf(self, N, name="pr", log_scale=False):
        """
        Reads results from output file

        Returns:
            Dictionary with parsed results
        """
        try:
            with open("new_output.txt", 'r', encoding='UTF-8') as file:
                lines = file.readlines()

                # Parse success and failure rates/times
                results = {}

                first_line = lines[0].split()
                for i, word in enumerate(first_line):
                    if word == "rate=":
                        results['success_rate'] = float(first_line[i+1])
                    elif word == "time=":
                        results['success_time'] = float(first_line[i+1])

                second_line = lines[1].split()
                for i, word in enumerate(second_line):
                    if word == "rate=":
                        results['failure_rate'] = float(second_line[i+1])
                    elif word == "time=":
                        results['failure_time'] = float(second_line[i+1])

                # Parse time distributions
                dt = []
                pTimeN = []
                pTime0 = []

                for line in lines[2:]:
                    l = line.split()
                    dt = np.append(dt, float(l[0]))
                    pTimeN = np.append(pTimeN, float(l[1]))
                    pTime0 = np.append(pTime0, float(l[2]))

            
                ptotal = pTime0 * results['failure_rate'] + \
                    pTimeN * results['success_rate']
                if log_scale==True:
                    ptotal = np.log(ptotal)
                ptotal = ptotal.reshape((len(ptotal), 1))

                results['dt'] = np.array(dt)
                results['pTimeN'] = np.array(pTimeN)
                results['pTime0'] = np.array(pTime0)
                results['ptotal'] = ptotal

                folder_path = "data"
                filename = name + f"_{N}" + "_out" + '.npz'

                path = os.path.join(folder_path, filename)

                os.makedirs(folder_path, exist_ok=True)

                np.savez_compressed(path, success_rate=results['success_rate'], failure_rate=results["failure_rate"], success_time=results["success_time"], failure_time=results["failure_time"], dt=results['dt'],
                         pTimeN=results['pTimeN'], pTime0=results['pTime0'], ptotal=results['ptotal'])

                return results

        except Exception as e:
            print(f"Error reading output file: {e}")

    def plot_profile(self, N, profile_type="linear", params=None):
        """
        Plot generated profile

        Args:
            profile_type: type of profile ('linear', 'small_min')

        """
        result = self.gen_profile(N, profile_type, params)

        fig, ax = plt.subplots()
        zn = result['dt']
        F = result['F']
        ax.plot(zn, F)

        plt.show()

    def plot_pdf(self, result, ref=None, name="pr"):
        """
        Plot time distributions for a given N

        Args:
            result: Result dictionary from fp calculation
            ref: Optional reference result to compare with
        """
        if not result:
            print("No data to display")
            return

        fig, ax = plt.subplots()

        folder_path = "data"
        os.makedirs(folder_path, exist_ok=True)
        filename = name + f"_{result['N']}" + "_out" + '.npz'
        path = os.path.join(folder_path, filename)

        try:
            with np.load(path) as data:
                dt = data['dt']
                total = data['ptotal']
                success = data['pTimeN']
                failure = data['pTime0']
        

            ax.plot(dt, total, 'b-', linewidth=2,
                    label=f"Total distribution (N={result['N']})")

            # Plot reference if provided
            if ref:
                ax.plot(ref['dt'], ref['ptotal'], 'g--',
                        label=f"(N_ref={ref['N']})")

            ax.legend(loc='best')
            plt.show()
        except FileNotFoundError:
                print(f"File {filename} not found!")



class BayesOptimizer:
    def __init__(self, solver, N_ref=100):
        """
        Args:
            solver: FokkerPlanckSolver instance
            N_ref: Reference N value for comparison
        """
        self.solver = solver
        self.N_ref = N_ref
        self.reference_model = None

        # Initialize log file
        with open("optimization_log.txt", "w") as f:
            f.write(f"Reference N: {N_ref}\n\n")

    def compute_reference_model(self):
        print(f"Computed reference model for N_ref={self.N_ref}")
        self.reference_model = self.solver.run_fp(self.N_ref)
        if self.reference_model:
            print(f"Reference model ptotal: {self.reference_model['ptotal']}")
            return True
        return False

    def loss_function(self, theta):
        """
        Loss function for Bayesian optimization

        Args:
            theta: array-like parameter

        Returns:
            Loss value based on difference from reference model
        """
        if self.reference_model is None:
            print("ERROR: Target not computed!")

        N_value = int(theta[0])

        # Run Fortran program for this N value
        result = self.solver.run_fp(N_value)

        current_value = result['ptotal']
        target = self.reference_model['ptotal']

        difference = np.mean((current_value - target)**2)  # mse

        # Write to log file
        with open("optimization_log.txt", "a") as f:
            f.write(f"N={N_value}, Loss={difference}\n")

        return np.array([[difference]])

    def run_bayesopt(self, max_iter=5, initial_points=50, domain=(2, 500), plot_results=True):
        """
        Run Bayesian optimization

        Args:
            max_iter: Maximum number of iterations
            initial_points: Number of initial random points
            domain: Range of N values to consider
            plot_results: Whether to plot results after optimization
        """
        if self.reference_model is None:
            success = self.compute_reference_model()
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

        # Select randomly 50 points to use as the initial training set
        feasible_region = GPyOpt.Design_space(space=space)
        initial_design = GPyOpt.experiment_design.initial_design(
            "random", feasible_region, initial_points)

        # Setup optimizer components
        objective = GPyOpt.core.task.SingleObjective(self.loss_function)

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

        for i in range(10):
            try:
                bo.run_optimization(
                    max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

                best_N = bo.x_opt[0]
                opt_loss = bo.fx_opt

                # Print the current best result
                print(f"Iteration: {(i + 1) * 5}")
                print(f"Best N value: {best_N}")
                print(f"Objective function value: {opt_loss}")

                # Check if the best N is the reference N
                if int(best_N) == self.N_ref:
                    print(
                        f"Optimization stopped: Best N={best_N} equals N_ref={self.N_ref}")
                    best_result = self.reference_model
                    break

                best_result = self.solver.run_fp(best_N)

                if plot_results:
                    self.solver.plot_pdf(
                        best_result, self.reference_model)

                input("Press Enter to continue...")

            except Exception as e:
                print(f"Error in optimization iteration {i+1}: {e}")
                traceback.print_exc()
                return None

        return {
            'best_N': best_N,
            'best_loss': opt_loss,
            'bo_object': bo,
            'final_result': best_result

        }


# Example usage
if __name__ == "__main__":  # Preventing unwanted code execution during import
    # Initialize solver
    solver = FokkerPlanckSolver()

    # Initialize optimizer
    N_0 = 100
    optimizer = BayesOptimizer(solver, N_ref=N_0)

    # Compute reference model
    optimizer.compute_reference_model()

    # Plot reference model profile and distribution
    solver.plot_profile(N_0)
    solver.plot_pdf(optimizer.reference_model)

    # Run optimization
    result = optimizer.run_bayesopt()

    if result:
        best_N = result['best_N']
        print(f"\n Best N is: {best_N}")

        result['bo_object'].plot_convergence()

        # Compare optimal and reference distributions
        fig, ax = plt.subplots()
        ax.plot(optimizer.reference_model['dt'], optimizer.reference_model['ptotal'],
                'b-', label=f'N_0={N_0})')
        ax.plot(result['final_result']['dt'], result['final_result']['ptotal'],
                'r-', label=f'Best N (N={best_N})', linestyle="--")

        plt.legend()
        plt.show()

    npz_files = glob.glob(os.path.join("data", '*.npz'))

    for file in npz_files:
        os.remove(file)
