#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to solve the Alternating Binary Sequence Problem (ABSP) via optimization

( https://www.geeksforgeeks.org/number-flips-make-binary-string-alternate/ )

When: Fri Jul 15 15:16:16 2022
"""
import os
import sys
from time import time, ctime

import picos
import numpy


SOLVER='glpk'

class Instance ():
    """Class that represents an ABSP instance."""

    def __init__ (self,n,A):
        self.n = n
        self.A = A
        
    def __str__(self):
        return "n = {0}, A = {1}".format(self.n,self.A)


def solveABSP (instance: Instance):
    """
        Solve the passed ABSP instance in three steps:
            1. Create the mathematical program;
            2. Run the solver;
            3. Report the results.
    """

    # create the MP
    A = instance.A
    N = range(instance.n)

    mp = picos.Problem()

    x = picos.BinaryVariable('x', instance.n)
    y = picos.BinaryVariable('y', instance.n)
    z = picos.BinaryVariable('z', instance.n)

    mp.set_objective('min', picos.sum([x[i] for i in N]))

    for i in N:
        mp.add_constraint(y[i] >= A[i] - x[i])
        mp.add_constraint(y[i] >= -A[i] + x[i])
        mp.add_constraint(y[i] <= A[i] - x[i] + 2*(1-z[i]))
        mp.add_constraint(y[i] <= -A[i] + x[i] + 2*z[i])
        mp.add_constraint(y[i] >= 0)
        if i < (instance.n-1):
            mp.add_constraint(y[i] + y[i+1] == 1)

    # solve the MP
    start = time()
    mp.solve(solver=SOLVER,verbosity=False)
    end = time()

    # report the results
    print("\nRESULTS\n-------")
    print("Status:  {0}".format(mp.status))
    print("ObjFun:  {0}".format(int(mp.value)))
    print("Input:   {0}".format(A))
    print("Flips:   {0}".format(
        list(map(int,numpy.array(x.value).T[0].tolist())))
        )
    print("Output:  {0}".format(
        list(map(int,numpy.array(y.value).T[0].tolist())))
        )
    print("Runtime: {0:.2f} seconds.\n".format(end-start))


def convertToInstance (arg: str) -> Instance :
    """
        Convert the input argument to an ABSP instance.
    
        Return None if the argument is not a sequence of bits.
    """

    n = int(len(arg.strip()))
    A = []
    for i in range(n):
        try:
            bit = int(arg[i].strip())
            if bit not in [0,1]:
                raise ValueError
            else:
                A.append(bit)
        except ValueError:
            print("Warning: input binary digits only, please.")
            return None
    return Instance(n, A)



if __name__ == "__main__":
    print("{0}: Starting execution of script {1}.\n".format(
        ctime(),
        os.path.basename(__file__)
        )
    )
    start = time()
    
    if len(sys.argv) == 2:
        instance = convertToInstance(sys.argv[1])
        if instance != None:
            solveABSP(instance)
    else:
        print("Usage: python alt_bin_seq_problem.py <binary sequence>")
    
    end = time()
    print("\n{0}: End of execution; time elapsed: {1:.2f} seconds.".format(
        ctime(),
        end-start
        )
    )