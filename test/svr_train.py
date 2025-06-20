import os,sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from model.SVRModel import SVR_Train
from scripts.run_proms import parse_args

def SVRTrain():
    args = parse_args()
    name = args.name
    is_scale = args.scale
    l = args.l
    r = args.r

    SVR_Train(name, is_scale, l, r)
    print("SVR Train Done!")

if __name__ == "__main__":
    SVRTrain()