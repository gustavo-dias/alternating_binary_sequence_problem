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
import picos

from time import time, ctime
from abc import ABC, abstractmethod
from enum import IntEnum



SOLVER='glpk'

class Instance ():
    """Class that represents ABSP instances."""

    def __init__ (self, size : int, sequence : list):
        self.size = size
        self.sequence = sequence

    @classmethod
    def create_from_string (cls, arg : str):
        """
            Create an ABSP instance or return None if arg is not a string \
            containing exclusevily zeros and/or ones.
        """
        size = int(len(arg.strip()))
        try:
            sequence = [int(bit) for bit in list(arg.strip())]
            for i in range(size):
                if sequence[i] not in [0,1]:
                    raise ValueError
        except ValueError:
            print("Warning: please input binary digits only.\n")
            return None
        return cls(size, sequence)
        
    def __str__(self):
        return "size = {0}, sequence = {1}".format(self.size, self.sequence)


class Solution ():
    """Class that represents solutions to the ABSP."""
    
    def __init__(self, 
                 variables : list,
                 num_flips: int = 0, 
                 runtime: float = 0.0,
                 ):
        self.num_flips = num_flips
        self.runtime = runtime
        self.flips = variables[0]
        self.output = variables[1]

    @property
    def report(self):
        print("\nRESULTS\n-------")
        print("NumFlips:  {0}".format(self.num_flips))
        print("Flips:     {0}".format(self.flips))
        print("Output:    {0}".format(self.output))
        print("Runtime:   {0:.2f} seconds.\n".format(self.runtime))


class AlgorithmType (IntEnum):
    """Class that implements an enumeration of types of algorithms."""
    
    b_and_b = 1
    algo_on = 2


class Algorithm (ABC):
    """Abstract class that represents solution algorithms."""
    
    @abstractmethod
    def run ():
        pass


class Problem ():
    """Class that represents ABSP problems."""
    
    def __init__(self, instance : Instance):
        self.instance = instance
        self.solution = None
 
    @property
    def mathprog_model (self):

        A = self.instance.sequence
        n = self.instance.size
        N = range(n)
    
        mp = picos.Problem()
    
        x = picos.BinaryVariable('x', n)
        y = picos.BinaryVariable('y', n)
        z = picos.BinaryVariable('z', n)
    
        mp.set_objective('min', picos.sum([x[i] for i in N]))
    
        for i in N:
            mp.add_constraint(y[i] >= A[i] - x[i])
            mp.add_constraint(y[i] >= -A[i] + x[i])
            mp.add_constraint(y[i] <= A[i] - x[i] + 2*(1-z[i]))
            mp.add_constraint(y[i] <= -A[i] + x[i] + 2*z[i])
            mp.add_constraint(y[i] >= 0)
            if i < (n-1):
                mp.add_constraint(y[i] + y[i+1] == 1)

        return mp

    def solve_using (self, algo_type : AlgorithmType, report : bool = False):
        """
            Solve the ABSP problem using algo_type; prints out the solution if \
            report is set to true.
        """
        if algo_type == AlgorithmType.b_and_b:
            print("Solving with B&B...")
            self.solution = BranchAndBound(self.mathprog_model, SOLVER).run()
        elif algo_type == AlgorithmType.algo_on:
            print("Solving with O(n)...")
            self.solution = AlgoON(self.instance).run()
        else:
            print("No algorithm selected.\n")
            
        if report and self.solution is not None:
            self.solution.report


class BranchAndBound (Algorithm):
    """
        Class that represents Branch and Bound algorithms.
    
        The class does not implement a B&B algorithm per see, it rather calls \
        third-party implementations.
    """
    
    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
    
    def run (self):
        """Run the B&B algorithm and return a Solution."""
        
        try:
            sol = self.model.solve(solver=self.solver, verbosity=False)
            
            float_values = sol.primals.values()
            int_values = [
                [int(bit) for bit in entry] for entry in list(float_values)
                ]
            
            return Solution(int_values, int(sol.value), sol.searchTime)

        except Exception as e:
            print("ERROR: when running B&B; returning 0-valued solution.")
            print(e)
            return Solution(0,0,[[],[]])


class AlgoON (Algorithm):
    """CLass that implements the O(n) algorithm."""
    
    def __init__(self, instance: Instance):
        self.instance = instance

    @staticmethod
    def get_alternating_sequences_of_size(arg : int):
        """Return lists storing the two alternating sequences of size arg."""
        
        seq_1, seq_2 = [], []
        
        for i in range(arg):
            if i % 2 == 0:
                seq_1.append(0)
                seq_2.append(1)
            else:
                seq_1.append(1)
                seq_2.append(0)
                
        return seq_1, seq_2
    
    
    def run (self):
        """Run the algorithm O(n) and return a Solution."""
        
        start = time()
    
        seq_1, seq_2 = self.get_alternating_sequences_of_size(
            self.instance.size
            )
        flips_to_seq_1, flips_to_seq_2 = [], []
        
        # loop over each of the input sequence's n bits, thus O(n)
        for i in range(self.instance.size):
            # compare with both sequences and store the map of required flips;
            # since seq_1 and seq_2 are distinct alternating sequences, for any
            # given sequence, it holds that:
            #   (P1) if sequence[i] != seq_1[i] then sequence[i] == seq_2[i];
            #   (P2) if sequence[i] == seq_1[i] then sequence[i] != seq_2[i].
            if self.instance.sequence[i] != seq_1[i]:
                # need to flip to reach seq_1
                flips_to_seq_1.append(1)
    
                # no need to flip to reach seq_2
                flips_to_seq_2.append(0)
            else:
                # no need to flip to reach seq_1
                flips_to_seq_1.append(0)
     
                # need to flip to reach seq_2
                flips_to_seq_2.append(1)

        # select solution based on minimum number of required flips, given by 
        # the sum of the entries in the lists flips_to_seq_#, where # in {1,2}.
        if sum(flips_to_seq_1) <= sum(flips_to_seq_2):
            solution = Solution([flips_to_seq_1, seq_1], sum(flips_to_seq_1))
        else:
            solution = Solution([flips_to_seq_2, seq_2], sum(flips_to_seq_2))
        
        end = time()
        solution.runtime = end-start
        return solution



if __name__ == "__main__":
    print("{0}: Starting execution of script {1}.\n".format(
        ctime(),
        os.path.basename(__file__)
        )
    )
    start = time()
    
    if len(sys.argv) == 2:
        instance = Instance.create_from_string(sys.argv[1])
        if instance != None:
            ABSP = Problem(instance)
            ABSP.solve_using(AlgorithmType.b_and_b, True)
            ABSP.solve_using(AlgorithmType.algo_on, True)
    else:
        print("Usage: python alt_bin_seq_problem.py <binary sequence>\n")
    
    end = time()
    print("{0}: End of execution; time elapsed: {1:.2f} seconds.".format(
        ctime(),
        end-start
        )
    )