
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

REQUIRED_VARS = ["BASE_PROJECT_PATH", "PYTHON_ENV"]


def load_and_validate_env(env_path=".env"):
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"File {env_path} is not found. Creating a template...")
        create_env_template(env_path)
        raise SystemExit(f"Fill in {env_path} and restart.")

    load_dotenv(dotenv_path=env_file)

    missing = [var for var in REQUIRED_VARS if os.getenv(var) is None]
    if missing:
        raise SystemExit(f"No variables in .env: {', '.join(missing)}")
    
def get_base_path():
    return os.getenv("BASE_PROJECT_PATH", "/tmp/translocation-task")

def create_env_template(path=".env"):
    content = """# .env
BASE_PROJECT_PATH=/data1/val2204
PYTHON_ENV=.venv
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Template {path} is created.")

