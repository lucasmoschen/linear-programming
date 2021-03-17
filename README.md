# Linear and Integral Programming 

Class material with examples and exercises about Linear and Integral
Programming, an area from Optimization. An optimization problem is, usually,
written as 

$$
\min f(x) \text{ subject to } x \in D, 
$$
where $D \subset \mathbb{R}^n$ is the feasible set and $f: D \to \mathbb{R}$
is the objective function. In this case, we are dealing with linear
programming, so the function $f$ will be an affine function, that is, if $q
\in \mathbb{R}^n$ and $c \in \mathbb{R}$, one can write $f(x) = q^Tx + c$ and
$D$ will be built from linear constraints, that is, 
$$
Ax \le b
$$
where $A \in \mathbb{R}^{m\times n}$ and $b \in \mathbb{R}^m$. Here the
inequality signal means pointwise comparison. And we can also put box
constraints like $x \ge 0$. 

Here I will put some problem sets written majorly in portuguese from this specific
subject and solve it manually and with Julia Optimization Package
[(JuMP)](https://jump.dev/JuMP.jl/stable/). I will use the GLPK optimizer for
the solvers and the structure will be as a notebook ([Jupyter
Notebook](https://jupyter.org/)) 

## Pset 1 

This is based on the first chapter of [A Gentle Introduction to
Optimization](https://www.amazon.com.br/Gentle-Introduction-Optimization-B-Guenin/dp/1107053447)
and it's focused on modelling problems. Here we will also have constraints of
the type $x_i$ is an integer. I could solve problems as sudoku with those
tools. 

## Pset 2 

## Pset 3