import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from utils.path_utils import get_output_folders

data_folder, images_folder = get_output_folders()

class FokkerPlanckSolver:
    def __init__(self):
        '''

        timesteps (Nt): actual number of timesteps
        dt: timestep (conversion into real units discussed)

        '''

        # Input file parameters
        self.timesteps = 100000
        self.dt = 0.01
        self.timedisplay = 10000  # timedisplay = Nt
        self.intdisplay = 10

        self.compile_fortran()

    def compile_fortran(self):
        """Compiles the Fortran program"""

        source_file = os.path.join(os.path.dirname(__file__), "fp.f90")
        output_path = os.path.join(data_folder, "output")
        try:
            compile_process = subprocess.run(
                ["gfortran", source_file, "-o", output_path],
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
            profile_type: type of profile ('linear', ...)

        Returns:
            Profile value at position i
        """

        if params is None:
            params = {}

        if profile_type == "linear":
            slope = params.get('slope', -0.01)
            c = params.get('c', 0)
            return slope * i + c

        elif profile_type == "gauss":
            A = params.get('A', -1)
            mu = params.get('mu', 20)
            sigma = params.get('sigma', 3)
            k = params.get('k', 0)
            return i*k + A * np.exp(-((i - mu)**2) / (2 * sigma**2))

        # When A == 0, turns into linear

        else:
            return -i/100  # Default linear profile

    def gen_profile(self, N, profile_type="linear", params=None):
        """
        Generates the input profile for a given N, makes an input file. 

        Args:
            N: scalar
                actual number of segments

            profile_type: str
                type of profile ('linear', )

            z: scalar
                actual length of the field (related to dt and the mobility constant)

        """
        N = int(N)
        z = N

        dt = []
        F = []

        path = os.path.join(data_folder, "new_input.txt")

        with open(path, "w", encoding="UTF-8") as fin:
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

    def run_fp(self, N, profile_type="linear", params=None, name="pr", 
               # log_scale=False, 
               exp_id=None):
        """
        Runs Fokker-Planck program with specified parameters

        Args:
            N: scalar
                actual number of segments
            profile_type: type of profile ('linear', 'gauss')

        Returns:
            Dictionary with simulation results
        """

        # Generate input profile
        self.gen_profile(N, profile_type, params)

        path = os.path.join(data_folder, "output")

        # Run Fortran program
        try:
            subprocess.run(
                [path],
                capture_output=True,
                text=True,
                cwd=data_folder
            )
            print(f"Calculation for profile : {profile_type}, N={N}, params: {params} completed!")

        except subprocess.CalledProcessError as e:
            print(f"Error running program for profile : {profile_type}, N={N}, params: {params}: {e}")
            print(f"Stderr: {e.stderr}")
            return None

        # Read results
        result = self.read_pdf(N, name, exp_id, 
                               # log_scale
                               )
        if result:
            result['N'] = int(N)
            result['profile_type'] = profile_type
            result['profile_params'] = params

        return result

    def read_pdf(self, N, name="pr", exp_id=None, 
                 # log_scale=False
                 ):
        """
        Reads results from output file

        Returns:
            Dictionary with parsed results
        """

        data_path = os.path.join(data_folder, "new_output.txt")

        try:
            with open(data_path, 'r', encoding='UTF-8') as file:
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

                if np.isnan(pTimeN).any():
                    raise ValueError(
                        f"NaN values detected in pTimeN for profile {name} with N={N}")
                    return None
                elif np.isnan(pTime0).any():
                    raise ValueError(
                        f"NaN values detected in pTime0 for profile {name} with N={N}")
                    return None

                ptotal = pTime0 * results['failure_rate'] + \
                    pTimeN * results['success_rate']
                # if log_scale == True:
                    # ptotal = np.log(ptotal)
                ptotal = ptotal.reshape((len(ptotal), 1))

                results['dt'] = np.array(dt)
                results['pTimeN'] = np.array(pTimeN)
                results['pTime0'] = np.array(pTime0)
                results['ptotal'] = ptotal

                if exp_id is not None:
                    filename = name + f"_exp{exp_id}_N{N}_out.npz"
                else:
                    filename = name + f"_N{N}_out.npz"


                filepath = os.path.join(data_folder, filename)

                np.savez_compressed(filepath, success_rate=results['success_rate'], failure_rate=results["failure_rate"], success_time=results["success_time"], failure_time=results["failure_time"], dt=results['dt'],
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

        filename = name + f"_N{result['N']}" + "_out" + '.npz'
        path = os.path.join(data_folder, filename)

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