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
from abc import ABC, abstractmethod
from enum import IntEnum

import picos


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
        return f"size = {self.size}, sequence = {self.sequence}"


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
        print(f"NumFlips:  {self.num_flips}")
        print(f"Flips:     {self.flips}")
        print(f"Output:    {self.output}")
        print(f"Runtime:   {self.runtime:.2f} seconds.\n")


class AlgorithmType (IntEnum):
    """Class that implements an enumeration of types of algorithms."""

    B_AND_B = 1
    ALGO_ON = 2


class Algorithm (ABC):
    """Abstract class that represents solution algorithms."""

    @abstractmethod
    def run (self):
        pass


class Problem ():
    """Class that represents ABSP problems."""

    def __init__(self, instance : Instance):
        self.instance = instance
        self.solution = None

    @property
    def mathprog_model (self):
        param_a = self.instance.sequence
        size = self.instance.size
        set_n = range(size)

        model = picos.Problem()

        var_x = picos.BinaryVariable('x', size)
        var_y = picos.BinaryVariable('y', size)
        var_z = picos.BinaryVariable('z', size)

        model.set_objective('min', picos.sum([var_x[i] for i in set_n]))

        for i in set_n:
            model.add_constraint(var_y[i] >= param_a[i] - var_x[i])
            model.add_constraint(var_y[i] >= -param_a[i] + var_x[i])
            model.add_constraint(
                var_y[i] <= param_a[i] - var_x[i] + 2*(1-var_z[i])
            )
            model.add_constraint(
                var_y[i] <= -param_a[i] + var_x[i] + 2*var_z[i]
            )
            model.add_constraint(var_y[i] >= 0)
            if i < (size-1):
                model.add_constraint(var_y[i] + var_y[i+1] == 1)

        return model

    def solve_using (self, algo_type : AlgorithmType, report : bool = False):
        """
            Solve the ABSP problem using algo_type; prints out the solution if \
            report is set to true.
        """
        if algo_type == AlgorithmType.B_AND_B:
            print("Solving with B&B...")
            self.solution = BranchAndBound(self.mathprog_model, SOLVER).run()
        elif algo_type == AlgorithmType.ALGO_ON:
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

        except Exception as e_exc:
            print("ERROR: when running B&B; returning 0-valued solution.")
            print(e_exc)
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


def main () -> None:
    """Run the script in full."""
    print(f"{ctime()}: Starting execution of script {os.path.basename(__file__)}.\n")
    start = time()

    if len(sys.argv) == 2:
        instance = Instance.create_from_string(sys.argv[1])
        if instance is not None:
            absp = Problem(instance)
            absp.solve_using(AlgorithmType.B_AND_B, True)
            absp.solve_using(AlgorithmType.ALGO_ON, True)
    else:
        print("Usage: python alt_bin_seq_problem.py <binary sequence>\n")

    end = time()
    print(f"{ctime()}: End of execution; time elapsed: {end-start:.2f} seconds.")


if __name__ == "__main__":
    main()
