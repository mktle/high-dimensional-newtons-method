#!/usr/bin/env python 
import numpy 
import sys
import matplotlib
matplotlib.use("agg") # Change to "agg" to run on FRI
from pylab import *
import tsase
import ase
import math
from numpy.linalg import inv
#from scipy import linalg as LA


def mag(a):
        return numpy.linalg.norm(a)

def gradient_descent(alpha, CC):
    f = p.get_forces()
    step = 0 
    while mag(f) > CC:
        step += 1
        f = p.get_forces()
        f *= alpha
        p.set_positions(p.get_positions() + f)
        f = p.get_forces()
        #print 'step:',step,'forces:', mag(f),'PE:',p.get_potential_energy()
        if(step == 100000):
            return False
    return True

def plotResults(measurement, fileNums, yLab, figSave, name):
    figure()
    xlabel("File Number")
    ylabel(yLab)
    bar(fileNums, measurement, align="center" )
    title(name)
    savefig(figSave)

numSuccesses = 0.
numFailures = 0.
numCalls = 0.

for i in range(100):
    clusterName = "lj38-clusters/" + str(i) + ".con"
    p = tsase.io.read_con(clusterName)
    lj = tsase.calculators.lj(cutoff=3.5)
    p.center(50.0)
    p.set_calculator(lj)
    opt = ase.optimize.FIRE(p, logfile = None, maxmove = 0.2)
    opt.run(fmax = 0.01)
    print(mag(p.get_positions()))
    numCalls += lj.force_calls
    print("For file", i, "there were", lj.force_calls, "calls")
print("Average force calls:", numCalls / 100.)
sys.exit()
