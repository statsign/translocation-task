import numpy as np
import GPyOpt
import traceback

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



    def compute_reference_model(self, name="pr"):
        print(f"Computed reference model for N_ref={self.N_ref}")
        self.reference_model = self.solver.run_fp(self.N_ref, name=name)
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
            This function does mse for
            ptotal = pTime0 * failure_rate + pTimeN * success_rate
            pTime0: time distribution for failure
            pTimeN: time distribution for success
        """
        if self.reference_model is None:
            print("ERROR: Target not computed!")

        N_value = int(theta[0])

        # Run Fortran program for this N value
        result = self.solver.run_fp(N_value)

        current_value = result['ptotal']
        target = self.reference_model['ptotal']

        difference = np.mean((current_value - target)**2)  # mse



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
                "name": "N",
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