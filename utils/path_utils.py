import os
from utils.env_utils import get_base_path

# base = "/data1/val2204"

def get_output_folders(base=get_base_path()):

    '''
    Makes folders for data and images
    Returns paths if it is necessary
    '''
    
    job_id = os.getenv('SLURM_JOB_ID', 'local')

    data_folder = os.path.join(base, "data", f"job_{job_id}")
    images_folder = os.path.join(base, "images", f"job_{job_id}")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    return data_folder, images_folder
