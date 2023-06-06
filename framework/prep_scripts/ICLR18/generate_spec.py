import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils import dump_json

def main():
    data = {"env_name": "ICLR18",
            "performance_eval_tasks": []}    
    
    dump_json(data, 'specs/ICLR18_spec.json')
    
    
if __name__ == '__main__':
    main()