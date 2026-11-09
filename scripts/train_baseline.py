"""Train one declared baseline on the frozen prepared data."""
import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.common.train import main
if __name__=='__main__':main()
