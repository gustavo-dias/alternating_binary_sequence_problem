#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The module implements a performance comparison between Mathematical Programming 
and an O(n) algorithm when solving the Alternating Binary Sequence Problem 
(ABSP).


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


def solveON (instance: Instance):
    """Solve the ABSP using an O(n) algorithm."""

    start = time()

    sol1, sol2 = [], []
    flips_sol1, flips_sol2 = [], []
    num_flips_to_sol1, num_flips_to_sol2 = 0, 0
    
    # loop over each of the n bits of the input sequence, thus O(n)
    for i in range(instance.n):

        # create the two alternating binary solutions to compare with the input
        # sequence; 
        if i % 2 == 0:
            sol1.append(0)
            sol2.append(1)
        else:
            sol1.append(1)
            sol2.append(0)
        
        # compare with the first solution whilst storing the map of necessary 
        # flips
        if instance.A[i] != sol1[i]:
            num_flips_to_sol1 += 1
            flips_sol1.append(1)
        else:
            flips_sol1.append(0)

        # compare with the second solution whilst storing the map of necessary
        # flips
        if instance.A[i] != sol2[i]:
            num_flips_to_sol2 += 1
            flips_sol2.append(1)
        else:
            flips_sol2.append(0)
    
    # select solution based on minimum number of necessary flips
    if num_flips_to_sol1 <= num_flips_to_sol2:
        num_flips = num_flips_to_sol1
        flips = flips_sol1
        output = sol1
    else:
        num_flips = num_flips_to_sol2
        flips = flips_sol2
        output = sol2
    
    end = time()

    # report the results
    print("\nRESULTS O(n)\n------------")
    print("NumFlips: {0}".format(num_flips))
    print("Input:    {0}".format(instance.A))
    print("Flips:    {0}".format(flips))
    print("Output:   {0}".format(output))
    print("Runtime:  {0:.2f} seconds.\n".format(end-start))
        

def solveMP (instance: Instance):
    """Solve the ABSP using Mathematical Programming."""

    start = time()

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
    mp.solve(solver=SOLVER,verbosity=False)

    end = time()

    # report the results
    print("\nRESULTS MP\n----------")
    print("NumFlips:  {0} (Status: {1})".format(int(mp.value), mp.status))
    print("Input:     {0}".format(A))
    print("Flips:     {0}".format(
        list(map(int,numpy.array(x.value).T[0].tolist())))
        )
    print("Output:    {0}".format(
        list(map(int,numpy.array(y.value).T[0].tolist())))
        )
    print("Runtime:   {0:.2f} seconds.\n".format(end-start))


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
            solveMP(instance)
            solveON(instance)
    else:
        print("Usage: python alt_bin_seq_problem.py <binary sequence>")
    
    end = time()
    print("\n{0}: End of execution; time elapsed: {1:.2f} seconds.".format(
        ctime(),
        end-start
        )
    )