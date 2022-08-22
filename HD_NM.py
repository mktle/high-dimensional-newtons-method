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

# calculates full hessian matrix numerically; d represents the finite difference value 
def hessian(d):
    H = numpy.zeros((3*len(p.get_positions()),3*len(p.get_positions())))
    pos = p.get_positions()
    for i in range (len(pos)):
        for j in range (len(pos[0])):
            H_r = []
            pos[i][j] += d
            p.set_positions(pos)
            d1 = p.get_forces()
            pos[i][j] -= 2 * d
            p.set_positions(pos)
            d2 = p.get_forces()
            H_i = (d2 - d1)/2/d
            H_r = numpy.reshape(H_i,3*len(p.get_positions()))
            H[3*i +j] = H_r
            pos[i][j] += d
    return H
# calculates full hessian matrix numerically with the option to replace some row with identity matrix 
# atomsinhessian is the atoms included in the hessians  
# curvatureguess is your guess at the curvature when not numerically calculating second derivative
def hessian_compare(d,atomsinhessian,curvatureguess):
    H = numpy.zeros((3*len(p.get_positions()),3*len(p.get_positions())))
    pos = p.get_positions()
    for i in range (len(pos)):
        for j in range (len(pos[0])):
          if i in atomsinhessian:
            H_r = []
            pos[i][j] += d
            p.set_positions(pos)
            d1 = p.get_forces()
            pos[i][j] -= 2 * d
            p.set_positions(pos)
            d2 = p.get_forces()
            H_i = (d2 - d1)/2/d
            H_r = numpy.reshape(H_i,3*len(p.get_positions()))
            H[3*i +j] = H_r
            pos[i][j] += d
          else:
            H_r = numpy.zeros(3*len(p.get_positions()))
            H_r[3*i+j] = curvatureguess 
            H[3*i +j] = H_r
    # Below makes the Hessian symmetric
    for i in atomsinhessian:
        for j in range(3):
            H[:,int(3*i+j)] = H[int(3*i+j)]
    return(H)

# Get eigenvalues and eigenvectors from Hessian
def HessianEigenvalues(atomsinHessian,dr,curvatureguess):
 HM = hessian_compare(dr,atomsinHessian,curvatureguess)
 Eigenvalue,Eigenvector = numpy.linalg.eig(HM)
 return Eigenvalue,Eigenvector

# Newton's method decomposed to eigenvectors --> negative eigenvalues shifted by lamda
def Newtons_method1(CC,maxstep,atomsh,curvatureguess):
    f = p.get_forces()
    global e_val
    step = 0 
    while mag(f) > CC:
        step += 1
        e_val,e_vec = HessianEigenvalues(atomsh,0.00000001,curvatureguess)
        dr = numpy.zeros(len(p.get_positions())*3)
        f = numpy.reshape(p.get_forces(),3*len(p.get_forces()))
        lamda = numpy.real(numpy.amin(e_val)) - 1
        for i in range(len(e_val)):
            if e_val[i] < 0.0001:
                dr += numpy.dot(numpy.real(e_vec[:,i]),f)/(numpy.real(e_val[i])-lamda) * numpy.real(e_vec[:,i])
            else:
                dr += numpy.dot(numpy.real(e_vec[:,i]),f)/(numpy.real(e_val[i])) * numpy.real(e_vec[:,i])
        if mag(dr) > maxstep:
            dr = maxstep*dr/mag(dr)
        dr_reshape = numpy.reshape(dr, numpy.shape(p.get_positions()) )
        p.set_positions(p.get_positions()+dr_reshape)
        f = p.get_forces()
        #print 'step:',step,'forces:', mag(f),'PE:',p.get_potential_energy(), 'curv',numpy.amin(e_val),'r',mag(dr)
    for eigen in e_val:
        if(eigen < 0.0001):
            return False
    return True

def plotResults(measurement, fileNums, yLab, figSave, name):
    figure()
    xlabel("File Number")
    ylabel(yLab)
    bar(fileNums, measurement, align="center" )
    title(name)
    savefig(figSave)

totalCalls = 0.
numSuccesses = 0
numFailures = 0

# create list of atom numbers to include in hessian
atomsh = []
for j in range(30):
    atomsh = numpy.append(atomsh, j)

for i in range(100):
    # import atoms object
    clusterName = "lj38-clusters/" + str(i) + ".con"
    p = tsase.io.read_con(clusterName)
    lj = tsase.calculators.lj(cutoff=3.5)
    p.center(50.0)
    p.set_calculator(lj)
    # run NM
    worked = Newtons_method1(0.01,0.3,atomsh,1000)
    if(worked):
        numSuccesses += 1
        totalCalls += lj.force_calls
    else:
        numFailures += 1
    print("For file", i, "there were", lj.force_calls, "calls")
print("Success to failure ratio ---", numSuccesses, ":", numFailures)
print("Average force calls:", totalCalls / numSuccesses)
sys.exit()
