import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__) 


def get_project_root():
    current_file = Path(__file__).resolve()
    

    for parent in current_file.parents:
        if (parent / "profiles.json").exists() or \
        (parent / "setup.py").exists() or \
        (parent / ".git").exists() or \
        (parent / "src").exists():
            project_root = parent
            break  
    
    if project_root:
        return project_root
    else:
        raise FileNotFoundError("Project root is not found")


def get_json_path():
    project_root = get_project_root()
    json_path = project_root / "profiles.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"File profiles.json not found in {project_root}")
    
    return json_path


def get_output_folders(base=get_project_root()):

    '''
    Makes folders for data and images
    Returns paths if necessary
    '''

    job_id = os.getenv('SLURM_JOB_ID', 'local')

    data_folder = os.path.join(base, "data", f"job_{job_id}")
    images_folder = os.path.join(base, "images", f"job_{job_id}")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    return data_folder, images_folder