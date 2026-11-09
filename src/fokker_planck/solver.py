import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import subprocess
import os
from utils.path_utils import get_output_folders
import logging
logger = logging.getLogger(__name__) 

data_folder, images_folder = get_output_folders()

class FokkerPlanckSolver:
    PROFILE_DEFAULTS = {
        "linear": {"slope": -0.01, "c": 0},
        "gauss": {"A": -1, "mu": 20, "sigma": 5, "k": 0},
    }
    def __init__(self):
        '''
        timesteps (Nt): actual number of timesteps
        dt: timestep (conversion into real units discussed)
        timedisplay: number of timesteps displayed in output file
        intdisplay: output interval
        '''

        # Input file parameters
        self.timesteps = 100000
        self.dt = 0.01
        self.timedisplay = 100000  
        self.intdisplay = 1

        self.compile_fortran()

    def compile_fortran(self):
        """Compiles the Fortran program"""

        source_file = os.path.join(os.path.dirname(__file__), "fp.f90")
        output_path = os.path.join(data_folder, "output")

        if os.path.exists(output_path):
            os.remove(output_path)

        try:
            compile_process = subprocess.run(
                ["gfortran", source_file, "-o", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if compile_process.returncode == 0:
                logger.info("Successfull compilation\n")

            else:
                logger.error("Compilation error:")
                logger.error(compile_process.stderr)

        except Exception as e:
            logger.error(f"Compilation error: {e}")

    def profile_func(self, i, profile_type="linear", params=None):
            """
            Generates free energy profile shape

            Args:
                i: coordinate for each point
                profile_type: type of profile ('linear', 'gauss', ...)

            Returns:
                Profile value at position i
            """

            if params is None:
                params = {}

            if profile_type == "linear":
                p = {**self.PROFILE_DEFAULTS["linear"], **params}
                slope, c = p['slope'], p['c']
                return slope * i + c

            elif profile_type == "gauss":
                p = {**self.PROFILE_DEFAULTS["gauss"], **params}
                A, mu, sigma, k = p['A'], p['mu'], p['sigma'], p['k']
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
                type of profile ('linear', 'gauss', ...)

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
               exp_id=None,
               save=True):
        """
        Runs Fokker-Planck program with specified parameters

        Args:
            N: scalar
                actual number of segments
            profile_type: type of profile ('linear', 'gauss', ...)

        Returns:
            Dictionary with simulation results
        """

        # Generate input profile
        self.gen_profile(N, profile_type, params)

        path = os.path.join(data_folder, "output")

        # if Fortran fails, the old file must not
        # be read and saved under the name of the current profile
        output_file = os.path.join(data_folder, "new_output.txt")
        if os.path.exists(output_file):
            os.remove(output_file)

        proc = subprocess.run(
            [path],
            capture_output=True,
            text=True,
            cwd=data_folder
        )
        if proc.returncode != 0 or not os.path.exists(output_file):
            logger.error(f"Fortran failed for profile : {profile_type}, N={N}, params: {params} "
                         f"(return code {proc.returncode}): {proc.stderr.strip()}")
            return None
        logger.info(f"Calculation for profile : {profile_type}, N={N}, params: {params} completed!")

        # Read the results
        result = self.read_pdf(N, name, exp_id, 
                               save=save
                               )
        if result:
            result['N'] = int(N)
            result['profile_type'] = profile_type
            result['profile_params'] = params

        return result

    def read_pdf(self, N, name="pr", exp_id=None, 
                 save=True
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
                elif np.isnan(pTime0).any():
                    raise ValueError(
                        f"NaN values detected in pTime0 for profile {name} with N={N}")

                ptotal = pTime0 * results['failure_rate'] + \
                    pTimeN * results['success_rate']
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

                if save:
                    np.savez_compressed(filepath, success_rate=results['success_rate'],
                        failure_rate=results["failure_rate"], success_time=results["success_time"], 
                        failure_time=results["failure_time"], dt=results['dt'],
                        pTimeN=results['pTimeN'], pTime0=results['pTime0'], ptotal=results['ptotal'])
                    # makes an npz file with all the arrays for further usage

                return results

        except Exception as e:
            logger.error(f"Error reading output file: {e}")

    def plot_profile(self, N, profile_type="linear", params=None):
        """
        Plot generated profile

        Args:
            profile_type: type of profile ('linear', 'gauss', ...)

        """
        result = self.gen_profile(N, profile_type, params)

        fig, ax = plt.subplots()
        zn = result['dt']
        F = result['F']
        ax.plot(zn, F)

        # plt.show()

    def plot_pdf(self, result, ref=None, name="pr"):
        """
        Plot time distributions for a given N

        Args:
            result: Result dictionary from fp calculation
            ref: Optional reference result to compare with
        """
        if not result:
            logger.error("No data to display")
            return

        fig, ax = plt.subplots()

        filename = name + f"_N{result['N']}" + "_out" + '.npz' 
        # takes the results from .npz files
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
            # plt.show()
        except FileNotFoundError:
            logger.error(f"File {filename} not found!")