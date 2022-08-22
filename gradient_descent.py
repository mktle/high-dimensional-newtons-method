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


def HessianEigenvalues(atomsinHessian, dr, curvatureguess):
    HM = hessian_compare(dr, atomsinHessian, curvatureguess)
    Eigenvalue, Eigenvector = numpy.linalg.eig(HM)
    return Eigenvalue, Eigenvector

def hessian_compare(d, atomsinhessian, curvatureguess):
    H = numpy.zeros((3*len(p.get_positions()), 3*len(p.get_positions())))
    pos = p.get_positions()
    for i in range(len(pos)):
        for j in range(len(pos[0])):
            if i in atomsinhessian:
                H_r = []
                pos[i][j] += d
                p.set_positions(pos)
                d1 = p.get_forces()
                pos[i][j] -= 2 * d
                p.set_positions(pos)
                d2 = p.get_forces()
                H_i = (d2 - d1)/2/d
                H_r = numpy.reshape(H_i, 3*len(p.get_positions()))
                H[3*i+j] = H_r
                pos[i][j] += d
            else:
                H_r = numpy.zeros(3*len(p.get_positions()))
                H_r[3*i+j] = curvatureguess
                H[3*i+j] = H_r
    for i in atomsinhessian:
        for j in range(3):
            H[:, int(3*i+j)] = H[int(3*i+j)]
    return(H)

def mag(a):
        return numpy.linalg.norm(a)

def gradient_descent(alpha, CC, maxstep, atomsh, curvatureguess):
    f = p.get_forces()
    step = 0 
    newPos = p.get_positions()
    running = True
    while (mag(f) > CC and running):
        step += 1
        dr = p.get_forces() * alpha
        if(mag(dr) > maxstep):
            dr = maxstep*dr/mag(dr)
        p.set_positions(p.get_positions() + dr)
        f = p.get_forces()
        #print 'step:',step,'forces:', mag(f),'PE:',p.get_potential_energy()
        if(step == 1000000):
            return False
    print((mag(f)))
    e_val, e_vec = HessianEigenvalues(atomsh, 0.00000001, curvatureguess)
    for eigen in e_val:
        if(eigen + 0.0005 < 0):
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
atomsh = []
for j in range(30):
    atomsh = numpy.append(atomsh, j)

for i in range(100):
    print(i)
    clusterName = "lj38-clusters/" + str(i) + ".con"
    p = tsase.io.read_con(clusterName)
    lj = tsase.calculators.lj(cutoff=3.5)
    p.center(50.0)
    p.set_calculator(lj)
    worked = gradient_descent(0.001, 0.01, 0.2, atomsh, 1000)
    if(worked):
        numSuccesses += 1
        numCalls += lj.force_calls
    else:
        numFailures += 1
    print("For file", i, "there was", lj.force_calls, "calls")    
print("Average force calls:", numCalls / numSuccesses)
print("Success to failure ratio --  ", numSuccesses, ":", numFailures)
sys.exit()
