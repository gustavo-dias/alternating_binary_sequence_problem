#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alternating Binary Sequence Problem (ABSP).

The module implements a performance comparison between a Branch and Bound (B&B)
and an O(n) algorithm when solving the ABSP.

The B&B used is not explicitly implemented in this module but rather called
from a public third-party solver.

Usage
-----
    python3 absp.py <binary_sequence>

Arguments
---------
    binary_sequence : str
        The string containing the sequence of zeros and ones.

Output
------
    For each solution approach, a brief report containing:
        (O1) The minimum number of flips;
        (O2) The map of flips;
        (O3) The solver/algorithm runtime and;
        (O4) The closest alternating sequence flip-wise.
"""

import os
import sys

from time import time, ctime
from abc import ABC, abstractmethod
from enum import IntEnum

import picos


# solver providing the B&B implementation
SOLVER='glpk'


class Instance ():
    """Class that represents ABSP instances.

    Attributes
    ----------
        size : int
            The size of the instance.
        sequence : list
            The sequence of zeros and ones.
    """

    def __init__ (self, size : int, sequence : list):
        """
        Parameters
        ----------
            size : int
                The size of the instance.
            sequence : list
                The sequence of zeros and ones.
        """
        self.size = size
        self.sequence = sequence

    @classmethod
    def create_from_string (cls, arg : str):
        """Create an ABSP instance from string arg.

        Return None if arg is not a string containing exclusevily zeros and/or
        ones.
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
    """Class that represents solutions to the ABSP.

    Attributes
    ----------
        num_flips : int
            The number of required flips to obtain an alternating sequence.
        runtime : float
            The runtime of the solver/algorithm.
        flips : list
            The map of required flips to obtain an alternating sequence.
        output : list
            The closest alternating sequence flip-wise.
    """

    def __init__(self,
                 variables : list,
                 num_flips: int = 0,
                 runtime: float = 0.0,
                 ):
        """
        Attributes
        ----------
            variables : list
                A list containing both the map of required flips and the
                closest alternating sequence flip-wise.
            num_flips : int
                The number of required flips to obtain an alternating sequence.
            runtime : float
                The runtime of the solver/algorithm.
        """
        self.num_flips = num_flips
        self.runtime = runtime
        self.flips = variables[0]
        self.output = variables[1]

    @property
    def report(self):
        """Print out the solution report."""
        print("\nRESULTS\n-------")
        print(f"NumFlips:  {self.num_flips}")
        print(f"Flips:     {self.flips}")
        print(f"Output:    {self.output}")
        print(f"Runtime:   {self.runtime:.2f} seconds.\n")


class AlgorithmType (IntEnum):
    """Class that implements an enumeration of types of algorithms.

    Options
    -------
        B_AND_B : Branch and Bound.
        ALGO_ON : A trivial O(n) algorithm.
    """

    B_AND_B = 1
    ALGO_ON = 2


class Algorithm (ABC):
    """Abstract class that represents solution algorithms."""

    @abstractmethod
    def run (self):
        """Run the algorithm."""
        pass


class Problem ():
    """Class that represents ABSP problems.

    Attributes
    ----------
        instance : Instance
            The ABSP data instance to be solved.
        solution : Solution
            The solution to the problem's instance.
    """

    def __init__(self, instance : Instance):
        """
        Parameters
        ----------
            instance : Instance
                An ABSP data instance.
        """
        self.instance = instance
        self.solution = None

    @property
    def mathprog_model (self):
        """Get the mathematical programming model of the problem."""
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
        """Solve the ABSP problem using algo_type.

        Prints out the solution if argument report is set to true.
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
    """Class that represents Branch and Bound algorithms.

    The class does not implement a B&B algorithm per see, it rather calls
    third-party implementations.

    Attributes
    ----------
        model : picos.Problem
            The ABSP model to be solved via B&B.
        solver : str
            The name of the B&B implementation to be used.
    """
    def __init__(self, model, solver):
        """
        Parameters
        ----------
            model : picos.Problem
                The ABSP model to be solved via B&B.
            solver : str
                The name of the B&B implementation to be used.
        """
        self.model = model
        self.solver = solver

    def run (self):
        """Run the B&B algorithm.

        Return a 0-valued solution in case errors occur during B&B execution.
        """
        try:
            sol = self.model.solve(solver=self.solver, verbosity=False)

            float_values = sol.primals.values()
            int_values = [
                [int(bit) for bit in entry] for entry in list(float_values)
                ]

            return Solution(int_values, int(sol.value), sol.searchTime)

        except picos.SolutionFailure as sf_exc:
            print(f"{type(self).__name__}: B&B run failed due to " \
                  f"{type(sf_exc).__name__}:")
            print(f" -> {sf_exc}")
            print("Returning 0-valued solution.\n")
            return Solution(0,0,[[],[]])


class AlgoON (Algorithm):
    """CLass that implements the O(n) algorithm.

    Attributes
    ----------
        instance : Instance
            The ABSP data instance to be solved.
    """

    def __init__(self, instance: Instance):
        """
        Parameters
        ----------
            instance : Instance
                The ABSP data instance to be solved.
        """
        self.instance = instance

    @staticmethod
    def get_alternating_sequences_of_size(arg : int):
        """Return two lists storing the alternating sequences of size arg."""
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
        """Run the O(n) algorithm."""
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
    print(f"{ctime()}: Starting execution of {os.path.basename(__file__)}.\n")
    start = time()

    if len(sys.argv) == 2:
        instance = Instance.create_from_string(sys.argv[1])
        if instance is not None:
            absp = Problem(instance)
            absp.solve_using(AlgorithmType.B_AND_B, True)
            absp.solve_using(AlgorithmType.ALGO_ON, True)
    else:
        print("Usage: python3 absp.py <binary_sequence>\n")

    end = time()
    print(f"{ctime()}: Terminated; time elapsed: {end-start:.2f} seconds.")


if __name__ == "__main__":
    main()
