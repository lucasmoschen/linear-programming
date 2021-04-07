# Linear and Integral Programming 

Class material with examples and exercises about Linear and Integral
Programming, an area from Optimization. An optimization problem is, usually,
written as

```{math}
minimize f(x) subject to  x em D, 
```

where D, the feasible set, is subset of n-dimensional real space and f, objective function from D to the real line.
In this case, we are dealing with linear programming, so the  objective function will be an [affine function](https://en.wikipedia.org/wiki/Affine_transformation) and D will be built from linear constraints, that is, it will be a simplex. 

Here I will put some problem sets written majorly in portuguese from this specific
subject and solve it manually and with Julia Optimization Package
[(JuMP)](https://jump.dev/JuMP.jl/stable/). I will use the GLPK optimizer for
the solvers and the structure will be as a notebook ([Jupyter
Notebook](https://jupyter.org/)) 

## Assignments 

### Assignment 1 

## Problem Sets

### Pset 1 

This is based on the first chapter of [A Gentle Introduction to
Optimization](https://www.amazon.com.br/Gentle-Introduction-Optimization-B-Guenin/dp/1107053447)
and it's focused on modelling problems. Here we will also have constraints of
the type x is an integer. I could solve problems as sudoku with those
tools. 

### Pset 2 

This problem set is based on the eighth chapter from [Applied Mathematical
Programming](http://web.mit.edu/15.053/www/AMP.htm) and the questions are
related to the general network-flow problem. For instance, the transportation
problem, the assignment problem and the maximum flow problem. 

### Pset 3

This problem set is based on the ninth chapter from [Applied Mathematical
Programming](http://web.mit.edu/15.053/www/AMP.htm) and the questions are
related to the integer linear programming.
