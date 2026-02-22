from solver.fokker_planck import FokkerPlanckSolver
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.plot_utils import init_plot_style
from utils.path_utils import get_output_folders

data_folder, images_folder = get_output_folders()

class CompareProfiles:
    def __init__(self, profiles, log_scale=False, experiment_id=0):

        # Initialize solver
        self.solver = FokkerPlanckSolver()
        self.profiles = profiles
        self.log_scale = log_scale
        self.experiment_id = experiment_id

        

        init_plot_style()


    def run_multiple_simulations(self, N):
        self.plot_profiles(N)

        results = []
        for profile in self.profiles:
            result = self.solver.run_fp(
                N, profile_type=profile['type'], params=profile['params'],
                name=profile['name'],
                log_scale=self.log_scale, exp_id=self.experiment_id)
            if result:
                result['label'] = profile['label']
                result['name'] = profile['name']
                result['profile'] = profile
                results.append(result)

        self.plot_multiple_pdf(results, N)
        self.compare_multiple_pdf(results, N)

        self.plot_multiple_success(results, N)
        self.plot_multiple_failure(results, N)

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
        img_path = os.path.join(images_folder, imgname)
        plt.savefig(img_path)
        plt.close()

    def plot_multiple_pdf(self, results, N):
        if not results:
            print("No data to display")
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
                if self.log_scale == False:
                    axes[i].set_ylabel('p(t)')
                else:
                    axes[i].set_ylabel('p(t) log scale')
                    axes[i].set_yscale('log')
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

    def plot_multiple_success(self, results, N):
        if not results:
            print("No data to display")
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
                print(f"File {filename} not found!")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        ax.set_xlabel('t')
        if self.log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('p_T (log scale)')
        else:
            ax.set_ylabel('p_T')
            ax.legend()
        ax.legend()
        plt.tight_layout()
        imgname = f"success_pdfs_exp{self.experiment_id}_{N}"
        plt.savefig(os.path.join(images_folder, imgname))
        plt.close()

    def plot_multiple_failure(self, results, N):
        if not results:
            print("No data to display")
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
                print(f"File {filename} not found!")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        ax.set_xlabel('t')
        if self.log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('p_F (log scale)')
        else:
            ax.set_ylabel('p_F')
            ax.legend()
        ax.legend()
        plt.tight_layout()
        imgname = f"failure_pdfs_exp{self.experiment_id}_{N}"
        plt.savefig(os.path.join(images_folder, imgname))
        plt.close()

    def compare_multiple_pdf(self, results, N):
        if not results:
            print("No data to display")
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
                if self.log_scale == False:
                    ax.set_ylabel('p(t)')
                    ax.autoscale()
                else:
                    ax.set_ylabel('p(t) log scale')
                    ax.set_yscale('log')
                    ax.set_ylim(1e-15, 0)

                ax.legend()

            except FileNotFoundError:
                print(f"File {filename} not found!")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        plt.tight_layout()
        imgname = f"compare_ptotal_exp{self.experiment_id}_N{N}"
        img_path = os.path.join(images_folder, imgname)
        plt.savefig(img_path)
        plt.close()
