from fokker_planck.solver import FokkerPlanckSolver
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.plot_utils import init_plot_style
from src.utils.path_utils import get_output_folders
import logging
logger = logging.getLogger(__name__) 

data_folder, images_folder = get_output_folders()

class CompareProfiles:
    def __init__(self, profiles,  experiment_id=1):

        # Initialize solver
        self.solver = FokkerPlanckSolver()
        self.profiles = profiles
        self.experiment_id = experiment_id

        init_plot_style()


    def run_multiple_simulations(self, N):
        self.plot_profiles(N)
        results = []
        for profile in self.profiles:
            result = self.solver.run_fp(
                N, profile_type=profile['type'], params=profile['params'],
                name=profile['name'],
                exp_id=self.experiment_id)
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['profile'] = profile
                results.append(result)

        self.plot_multiple_pdf(results, N)
        # self.compare_multiple_pdf(results, N)

        # self.plot_multiple_success(results, N)
        # self.plot_multiple_failure(results, N)

        return results

    def plot_profiles(self, N):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
        axes = axes.flatten()
        for i, profile in enumerate(self.profiles):
            # Generate the profile first
            result = self.solver.gen_profile(
                N, profile['type'], profile['params'])
            zn = result['dt']
            F = result['F']

            axes[i].plot(zn, F, label=f"{profile['label']}")
            axes[i].legend(loc='best')
            axes[i].set_xlabel("s")
            axes[i].set_ylabel("F(s)/kT")
            axes[i].set_xlim(0, N+1)

        plt.tight_layout()
        imgname = f"profiles_exp{self.experiment_id}_{N}"
        img_folder = os.path.join(images_folder, "profile_plots", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def plot_multiple_pdf(self, results, N):
        if not results:
            logger.error("No data to display")
            return
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
        axes = axes.flatten()

        for i, result in enumerate(results):
            filename = f"{result['name']}_exp{self.experiment_id}_N{N}_out.npz"
            filepath = os.path.join(data_folder, filename)
            try:
                with np.load(filepath) as data:
                    dt = data['dt']
                    total = data['ptotal']
                    if len(total.shape) > 1:
                        total = total.flatten()
                    success = data['pTimeN']
                    failure = data['pTime0']

                axes[i].plot(dt, total, 'b-', linewidth=2,
                             label=f"{result['label']}")
                axes[i].set_xlabel('t')
                axes[i].set_ylabel('p(t) log scale')
                axes[i].set_yscale('log')
                axes[i].legend()

            except FileNotFoundError:
                logger.error(f"File {filename} not found!")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
        plt.tight_layout()

        imgname = f"pdfs_exp{self.experiment_id}_N{N}"
        img_folder = os.path.join(images_folder, "multiple_pdfs", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def plot_multiple_success(self, results, N):
        if not results:
            logger.error("No data to display")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for result in results:
            filename = f"{result['name']}_exp{self.experiment_id}_N{N}_out.npz"
            filepath = os.path.join(data_folder, filename)
            try:
                with np.load(filepath) as data:
                    dt = data['dt']
                    success = data['pTimeN']
                    if success.ndim > 1:
                        success = success.flatten()
                ax.plot(dt, success, linewidth=2, label=f"{result['label']}")
            except FileNotFoundError:
                logger.error(f"File {filename} not found!")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
        ax.set_xlabel('t')
        ax.set_yscale('log')
        ax.set_ylabel('p_T (log scale)')
        ax.legend()
        plt.tight_layout()
        imgname = f"success_pdfs_exp{self.experiment_id}_{N}"
        img_folder = os.path.join(images_folder, "success_pdfs", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def plot_multiple_failure(self, results, N):
        if not results:
            logger.error("No data to display")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for result in results:
            filename = f"{result['name']}_exp{self.experiment_id}_N{N}_out.npz"
            filepath = os.path.join(data_folder, filename)
            try:
                with np.load(filepath) as data:
                    dt = data['dt']
                    failure = data['pTime0']
                    if failure.ndim > 1:
                        failure = failure.flatten()
                ax.plot(dt, failure, linewidth=2, label=f"{result['label']}")
            except FileNotFoundError:
                logger.error(f"File {filename} not found!")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
        ax.set_xlabel('t')
        ax.set_yscale('log')
        ax.set_ylabel('p_F (log scale)')
        ax.legend()
        plt.tight_layout()
        imgname = f"failure_pdfs_exp{self.experiment_id}_{N}"
        img_folder = os.path.join(images_folder, "failure_pdfs", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def compare_multiple_pdf(self, results, N):
        if not results:
            logger.error("No data to display")
            return

        fig, ax = plt.subplots()

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
                ax.set_ylabel('p(t) log scale')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-15)

                ax.legend()

            except FileNotFoundError:
                logger.error(f"File {filename} not found!")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
        plt.tight_layout()
        imgname = f"compare_ptotal_exp{self.experiment_id}_N{N}"
        img_folder = os.path.join(images_folder, "compare_pdfs", f"exp{self.experiment_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, imgname)
        plt.savefig(img_path)
        plt.close()
