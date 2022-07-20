# Alternating Binary Sequence Problem (ABSP)

ABSP: What is the minimum number of flips required to turn a given sequence of bits into an alternating sequence of bits?

Example: Consider the binary sequence 111. Two flips (first and third bits) yield the alternating sequence 010; however, the alternating sequence 101 can be obtained with one single flip (second bit), thus being the answer to this instance of the ABSP.


# Optimization Model

Let n be a natural number such that N = {1,...,n}.

Let a binary sequence be represented by the vector A in {0,1}^n.

Let x in {0,1}^n be a binary decision variable such that:
 - x_i = 0 if a_i must retain its original value;
 - x_i = 1 if a_i must change its original value.

Let y in {0,1}^n be a binary decision variable that represents an alternating binary sequence.

This script solves the ABSP via the following nonlinear mathematical program:

minimize	sum_{i in N}{x_i}
subject to:	|a_i - x_i| = y_i,	for all i in N;
		y_i + y_{i+1} = 1,	for all i in N\{n};
		x,y in {0,1}^n.


#Usage

python alt_bin_str_problem.py <binary sequence>

E.g.: python alt_bin_str_problem.py 11100110010111.


#Output

Minimum number of flips;
Map of bits that were changed in the original sequence A;
Alternating sequence y.
