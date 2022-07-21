# Alternating Binary Sequence Problem (ABSP)

__ABSP: What is the minimum number of flips required to turn a given sequence of bits into an alternating sequence of bits?__

Example: Consider the binary sequence 111. Two flips (first and third bits) yield the alternating sequence 010; however, the alternating sequence 101 can be obtained with one single flip (second bit), thus being the answer to this instance of the ABSP.

There are O(n) algorithms available for the ABSP [1]. The script allows to assess how far worse is the performance of optimization timewise.

### Optimization Model

Let $i,n$ be natural numbers such that $i \in N = \\{1,...,n\\}$.

Let a binary sequence be represented by the vector $A \in \\{0,1\\}^n$.

Let $x \in \\{0,1\\}^n$ be a binary decision variable such that:
 - $x_i = 0$ if $A_i$ must retain its original value;
 - $x_i = 1$ if $A_i$ must change its original value.

Let $y \in \\{0,1\\}^n$ be a binary decision variable that represents an alternating binary sequence.

The script solves the ABSP via the following nonlinear mathematical program:

$minimize \quad \sum_\limits{i \in N}{x_i}$

$subject\ to \quad |A_i - x_i| = y_i, \qquad \forall i \in N;$

$\qquad\qquad\quad\ y_i + y_{i+1} = 1,	\qquad\ \forall i \in N-\\{n\\}$;

$\qquad\qquad\quad\ x,y \in \\{0,1\\}^n$.


### Environment and Usage

The script uses Picos [2] and GLPK [3] to write and solve the mathematical model introduced in the previous section.

It can be run as:

`python alt_bin_seq_problem.py <binary sequence>`

E.g.: `python alt_bin_seq_problem.py 11100110010111`.


### Output

The minimum number of flips;
	
The map $x$ of bits that were changed in the original sequence $A$;
	
The alternating sequence $y$;

The runtime.


### References

[1]: Alternating Binary Sequence Problem. URL: https://www.geeksforgeeks.org/number-flips-make-binary-string-alternate/

[2]: Guillaume Sagnol and Maximilian Stahlberg. PICOS: A Python interface to conic optimization solvers. Journal of Open Source Software, Vol 7, Number 70, February 2022. DOI: [10.21105/joss.03915](https://doi.org/10.21105/joss.03915).

[3]: GNU Linear Programming Kit. URL: https://www.gnu.org/software/glpk/

