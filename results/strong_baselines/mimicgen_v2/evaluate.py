"""Load pinned robomimic before simulation imports; shared evaluation unchanged."""
import runpy
from baselines.common.language import install_core

def main():
    install_core()
    runpy.run_module('baselines.strong.evaluate',run_name='__main__')

if __name__=='__main__':main()
