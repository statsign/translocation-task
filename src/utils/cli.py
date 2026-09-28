import argparse
import json
import os
import logging
logger = logging.getLogger(__name__) 

def parse_experiment_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--N",
            type=int,
            nargs="+",
            default=[50],
            help="Set the value of N (default: 50)"
        )

    parser.add_argument(
            "--log_scale",
            action="store_true", # boolean
            default=False,
            help="Use logarithmic scale"
        )
    
    parser.add_argument(
            "--max_iter",
            type=int,
            default=10
        )
    
    parser.add_argument(
            "--init_points",
            type=int,
            default=50
        )


    return parser.parse_args()


def load_profile_sets(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        profile_sets = json.load(f)

    if not isinstance(profile_sets, list):
        raise ValueError("No profiles in JSON file")
    
    logger.info(f"Loaded {len(profile_sets)} profile sets")
    for i, profiles in enumerate(profile_sets):
        logger.info(f"  Set ofр {i+1}: {len(profiles)} profiles")
        for j, profile in enumerate(profiles):
            logger.info(f"    - {profile.get('name', f'profile_{j}')}: {profile.get('type')}")
    
    return profile_sets


