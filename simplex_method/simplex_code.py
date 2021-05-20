#!/usr/bin/env python
# coding: utf-8

import numpy as np

class SimplexMethod: 
    """
    Maximize a linear objective function subject to linear equality and
    inequality constraints. It solves the problem in the form: 

    max c^T x 
    subject to
    A_eq x  =  b_eq
    A_ub x <=  b_ub
    l <= x <= u
    
    where x is a vector of variables of the problem, c, b_ub, b_eq, l and u 
    are arrays with dimension 1, and A_ub and A_eq are arrays with dimension
    2.
    
    Parameters
    ---------- 
    c : numpy array 1-D, ex.: c = np.array([1,2])
        The coefficients of the linear objective function to be maximized.
    A_ub (optional): numpy array 2-D, ex.: A_ub = np.array([[1,2],[3,4]])
        The inequality constraint matrix. It rows is an inequality equation.
    b_ub (optional): numpy array 1-D, ex.: b_ub = np.array([1,2])
        The inequality constraint vector. It indicates the upper bounds of
        each inequality restriction.
    A_eq (optional): numpy array 2-D, ex.: A_eq = np.array([[1,2],[3,4]])
        The equality constraint matrix. It rows is an equation.
    b_eq (optional): numpy array 1-D, ex.: b_eq = np.array([1,2])
        The equality constraint vector. 
    bounds (optional): list of 2-tuples.
        It indicates a sequence of [min, max] for each variable. If it is nor
        informed, we suppose it variable in [0, np.inf). If only a subset of
        the variables has different bounds compared to the default, you must
        inform even when the bounds are [0, np.inf).
    x0 (optional):  numpy array 1-D, ex.: c = np.array([1,2])
        Initial solution to the iteration. It must be a vertex of the simplex.

    Result
    ---
    After instanciating the method and calling the function optimization (the
    only one available), the result will be:
        x : 1-D array
            The solution vector.
        fun : float
            The optimal value.
        success : bool
            True when the algorithm has completed successfully.
        slack : 1-D array
            Slack variables to each inequality restriction (including the upper bounds)
        nit : int
            The number of iterations required.
        message : str
            A string descriptor of the algorithm status.
    """
    
    def __init__(self, c, **kwargs): 
        
        self.c = c
        c = c.reshape(-1,1)
        self.n_var = c.shape[0]
        A_ub = kwargs.get("A_ub", np.empty(shape=(0,self.n_var)))
        A_eq = kwargs.get("A_eq", np.empty(shape=(0,self.n_var)))
        b_ub = kwargs.get("b_ub", np.empty(shape=(0,1)))
        b_eq = kwargs.get("b_eq", np.empty(shape=(0,1)))
        # If bounds is an empty list, treating as [0, +inf].
        bounds = kwargs.get("bounds", [])
        # initial solution 
        self.x0 = kwargs.get("x0", None)
        
        b_ub = b_ub.reshape(-1,1)
        b_eq = b_eq.reshape(-1,1)
        
        self.nit = 0
        
        self._check_dimensions(c, A_ub, A_eq, b_ub, b_eq)
        self.variables_transformations = {v: [[v],0] for v in range(self.n_var)}
        
        c, A_ub, A_eq, b_ub, b_eq = self._bounds_handler(bounds, c, A_ub, A_eq, b_ub, b_eq)
        slack, A_ub, b_ub = self._add_slack_variables(A_ub, b_ub)
        
        # Joining all the equalities
        slack = np.vstack([slack, np.zeros((A_eq.shape[0], slack.shape[0]))])
        A = np.vstack([A_ub, A_eq])
        b = np.vstack([b_ub, b_eq])
        slack, A, b = self._force_b_positive(slack, A, b)
        self.slack_pos = A.shape[1]
        A = np.hstack([A, slack])
        A, b = self._assemble_basis(A,b,c)
    
        self._add_artificial_variables(A, b, c)
        
        self.art_pos = A.shape[1]

    def _check_dimensions(self, c, A_ub, A_eq, b_ub, b_eq): 
        """
        Check the dimensions of each input. It will return an Exception if
        some of then do not fit. 
        """
        if A_ub.shape[1] != self.n_var: 
            raise Exception("The number of columns of A_ub must be the number of elements of c.")
        elif A_eq.shape[1] != self.n_var:
            raise Exception("The number of columns of A_eq must be the number of elements of c.")
        elif b_ub.shape[0] != A_ub.shape[0]:
            raise Exception("The number of lines of A_ub must be the number of elements of b_ub.")
        elif b_eq.shape[0] != A_eq.shape[0]: 
            raise Exception("The number of lines of A_eq must be the number of elements of b_eq.")
        if self.x0 is not None: 
            if self.x0.shape[0] != self.n_var:
                raise Exception("The number of elements in x0 must be the number of elements of c.")

    def _aux_free_variable(self, A_bar, A, col, i): 
        """
        For each real variable, that is, when the lower bound is -np.inf, it
        adds a new variable: x = x' - x'', where x', x'' >= 0.
        """
        A_bar[:A.shape[0], col] = A[:,i]
        A_bar[:A.shape[0], col+1] = -A[:,i]
        return A_bar

    def _bounds_handler(self, bounds, c, A_ub, A_eq, b_ub, b_eq):
        """
        This function deals with the input bounds: 
        - For each real upper bound x <= u, it adds a new restriction.
        - For each real lower bound x >= l, it defines x' = x - l and keep
          this new variable. The transformation is saved in a dictionary. 
        - For each infinite lower bound, x = x' - x'', x',x'' >= 0. The
          transformation is saved in a dictionary.
        """
        l = np.array([b[0] for b in bounds], dtype = np.float64)
        u = np.array([b[1] for b in bounds], dtype = np.float64)
        if len(bounds) == 0: 
            return (c, A_ub, A_eq, b_ub, b_eq)
        elif len(bounds) < self.n_var: 
            raise Exception("You need to specify dimension of c bounds.")
        else:
            free_variables = sum(l == -np.inf)
            upper_bounds = sum(u < np.inf)
            # Allocating space in case of free variables
            A_ub_bar = np.zeros((A_ub.shape[0] + upper_bounds, A_ub.shape[1] + free_variables))
            b_ub_bar = np.zeros((A_ub.shape[0] + upper_bounds, 1))
            b_ub_bar[:A_ub.shape[0],0:1] = b_ub
            A_eq_bar = np.zeros((A_eq.shape[0], A_eq.shape[1] + free_variables))
            c_bar = np.zeros((c.shape[0] + free_variables, 1))
            
            col = 0
            lin = 0
            for i in range(self.n_var):
                if l[i] == -np.inf: 
                    A_ub_bar = self._aux_free_variable(A_ub_bar, A_ub, col, i)
                    A_eq_bar = self._aux_free_variable(A_eq_bar, A_eq, col, i)
                    c_bar = self._aux_free_variable(c_bar.transpose(), c.transpose(), col, i).transpose()
                    # Save the necessary transformation after
                    self.variables_transformations[i] = ([col, col+1], 0)
                    if u[i] < np.inf: 
                        A_ub_bar[A_ub.shape[0]+lin,col] = 1
                        A_ub_bar[A_ub.shape[0]+lin,col+1] = -1
                        b_ub_bar[A_ub.shape[0]+lin] = u[i]
                        lin+=1
                    col+=2
                else: 
                    A_ub_bar[:A_ub.shape[0],col] = A_ub[:,i]
                    A_eq_bar[:,col] = A_eq[:,i]
                    c_bar[col] = c[i]
                    
                    self.variables_transformations[i] = ([col], l[i])
                    if u[i] < np.inf: 
                        A_ub_bar[A_ub.shape[0]+lin,col] = 1
                        b_ub_bar[A_ub.shape[0]+lin] = u[i]
                        lin+=1
                    b_ub_bar = b_ub_bar - A_ub_bar[:,col:col+1]*l[i]
                    b_eq = b_eq - A_eq[:,i:i+1]*l[i]
                    col+=1
        return (c_bar, A_ub_bar, A_eq_bar, b_ub_bar, b_eq)

    def _add_slack_variables(self, A_ub, b_ub):
        """
        For each inequality restriction, it adds a slack variable. The result
        will be an identity matrix which will be concatenated with the matriz
        A_ub later. 
        """
        slack_var = np.eye(A_ub.shape[0])
        return (slack_var, A_ub, b_ub)

    def _force_b_positive(self, slack_var, A, b): 
        """
        This algorithm works only if all b values are positive. This function
        transforms each restriction in order to correct that. 
        """
        negative_const = np.where(b < 0)[0]
        A[negative_const] = -A[negative_const]
        slack_var[negative_const] = - slack_var[negative_const]
        b[negative_const] = -b[negative_const]
        
        return slack_var, A, b

    def _assemble_basis(self, A, b, c): 
        """
        It builds the first basis from the equality and inequality matrix, now
        transformed in a linear system Ax = b. The basis is formed by
        variables which appear in a unique restriction with positive
        coefficient. Each restriction has a basic variable. If some has not in
        this phase, we ass artificial variables.  
        """
        self.basis = {i: None for i in range(A.shape[0])}
        for j in range(A.shape[1]): 
            test = 0.0 if j >= self.slack_pos else c[j]
            if test == 0: 
                non_zeros = np.where(A[:,j] != 0)[0]
                if (non_zeros.shape[0] == 1) & (A[non_zeros[0], j] > 0):
                    self.basis[non_zeros[0]] = j
                    A[non_zeros[0], :] = A[non_zeros[0],:]/A[non_zeros[0],j]
                    b[non_zeros[0]] = b[non_zeros[0]]/A[non_zeros[0],j]
        return A, b

    def _add_artificial_variables(self, A, b, c): 
        """
        Add an artificial variable for each restriction with no basic
        variable. We add in the end of the A matrix. This function also builds
        the tableaux. 
        """
        art_var = 0
        self.art_constraints = []
        for i in self.basis.keys(): 
            if self.basis[i] is None: 
                self.basis[i] = A.shape[1] + art_var 
                self.art_constraints.append(i)
                art_var += 1
        artificial_variables = np.eye(art_var)

        self.table = np.zeros((A.shape[0] + 2, A.shape[1] + art_var + 1))
        self.table[:-2, :-art_var-1] = A
        self.table[self.art_constraints, -art_var-1:-1] = artificial_variables
        self.table[:-2, -1] = b.flatten()
        self.table[-2,:c.shape[0]] = c.flatten()

    def _change_of_variables(self, tb, r, s): 
        """
        This function operates the pivoting. It pick two variables, a
        non-basic and another basic and changes them. 
        """
        a_rs = tb[r,s]
        norm_line_r = tb[r,:]/a_rs
        tb = tb - np.outer(tb[:,s], tb[r,:])/a_rs
        tb[r,:] = norm_line_r
        self.basis[r] = s

        return tb

    def _iteration(self,tb, phase1=False): 
        """
        This is an iteration of the simplex method. 
        """
        while sum(tb[-1,:-1] > 0) > 0: 
            
            self.nit += 1

            s = np.argmax(tb[-1,:-1])
            if sum(tb[:-1,s] > 0) == 0: 
                return None

            positive_a = np.where(tb[:-1-phase1,s] > 1e-16)[0]
            r = positive_a[(tb[positive_a,-1]/tb[positive_a,s]).argmin()]
            tb = self._change_of_variables(tb, r, s)
            
        return tb 

    def _phase1(self, table): 
        """
        The phase 1 verifies if an auxiliary problem (which the sum of artificial
        variables forms the objective function) has a optimal value 0. If this
        is the case, all the artificial variables are removed because they
        always will have a zero value. If this is not the case, the problem is
        unfeasible.
        """
        new_objective = table[self.art_constraints, :].sum(axis=0)
        table[-1,:] = new_objective
        table[-1, self.art_pos:] = 0.0

        table = self._iteration(table, phase1 = True)
        
        if -table[-1,-1] < new_objective[-1]:
            return None
        else: 
            for (cons, basis) in self.basis.items(): 
                if basis >= self.art_pos: 
                    for j in range(self.art_pos):
                        if table[cons, j] != 0: 
                            table = self._change_of_variables(table, cons, j)
                            break
            # keep the artificial variables always zero. 
            table = np.hstack([table[:,:self.art_pos], table[:,[-1]]])
            table = table[:-1,:]
        return table   

    def _phase2(self, table):
        """
        It just calls an new iteration. 
        """
        return self._iteration(table)

    def _change_basis(self, tb):
        
        assert sum(self.x0 != 0) == self.table.shape[0] - 2

        non_basic_variables_initial = []
        basic_initial = list(range(len(self.x0)))
        for restriction, basic in self.basis.items(): 
            basic_initial.remove(basic)
            if self.x0[basic] > 0:
                continue
            else: 
                non_basic_variables_initial.append((restriction, basic))

        for variable in basic_initial:
            basic_initial.remove(variable)

        assert len(basic_initial) == len(non_basic_variables_initial)

        for i in range(len(basic_initial)): 
            tb = self._change_of_variables(tb, basic_initial[i], non_basic_variables_initial[i][0])

        return tb            

    def _results(self, tb): 
        """
        This function organizes the results transforming each variable to the
        specified by the user. 
        """
        x = np.zeros(self.n_var)
        y = np.zeros(self.slack_pos)
        slack = np.zeros(tb.shape[1]-1-self.slack_pos)
        for (cons, basis) in self.basis.items(): 
            if basis < self.slack_pos:
                y[basis] = tb[cons, -1] 
            else: 
                slack[basis-self.slack_pos] = tb[cons, -1]
        for i in range(self.n_var): 
            position = self.variables_transformations[i][0]
            if len(position) == 1:  
                x[i] = y[position[0]] + self.variables_transformations[i][1]
            else: 
                x[i] = y[position[0]] - y[position[1]] 

        fun = sum(self.c*x)
            
        return x, slack, fun

    def optimize(self):
        """
        Call this function to start the optimization process.
        """
        tb = np.array(self.table)

        tb = self._phase1(tb)
        if tb is None: 
            message = "The problem is infeasible."
            success = False
            fun = None
            x = None
            slack = None
        else: 
            if self.x0 is not None: 
                tb = self._change_basis(tb)
            tb = self._phase2(tb)
            if tb is None: 
                message = "The primal problem is unbounded."
                success = False
                fun = None
                x = None
                slack = None
            else: 
                message = "Optimization terminated successfully."
                success = True
                x, slack, fun = self._results(tb)

        return OptimizeResult(x, fun, slack, success, message, self.nit)

class OptimizeResult: 
    """
    Represents the optimization function.
    """
    def __init__(self, x, fun, slack, success, message, nit): 
        
        self.x = x
        self.fun = fun
        self.slack = slack
        self.success = success
        self.message = message
        self.nit = nit
        
        self.result = {"x": x, "fun": fun, "slack": slack, "success": success, "message": message, "nit": nit}