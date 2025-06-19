import os,sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from model.LinearModel import Linear_Train
from scripts.run_proms import parse_args

def LinearTrain():
    args = parse_args()
    name = args.name
    is_scale = args.scale
    l = args.l
    r = args.r

    Linear_Train(name, is_scale, l, r)
    print("Linear Train Done!")

if __name__ == "__main__":
    LinearTrain()