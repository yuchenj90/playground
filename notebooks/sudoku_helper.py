import pulp as p 
import numpy as np
from termcolor import colored

def sudoku_read(filepath):
    # read input and return data that records position of each number
    # data is a list, each element is a list [v, r, c], corresponds to a value v occurred at row r and column c
    sudokuin = open('data/sudokuin.txt','r')
    l = sudokuin.readlines()
    num = [str(i) for i in range(1,10)]
    data = []
    for r in range(9):
        if r in [0, 3, 6]:
            print("+-------+-------+-------+")
        for c in range(9):
            if l[r][c] in num:
                data.append([int(l[r][c]),r+1,c+1])
            if c in [0, 3, 6]:
                print("| ", end ="")
            print(l[r][c]+" ",end="")
            if c == 8:
                print("|\n",end="")
    print("+-------+-------+-------+")
    return data

def sudoku_solver(input_data):
    '''
        Solve sudoku via LP formulation
        input_data is a list has following format:
            each element is a list [v, r, c], corresponds to a value v occurred at row r and column c
    '''
    Rows = [[(i,j) for j in range(1,10)] for i in range(1,10)]
    Cols = [[(i,j) for i in range(1,10)] for j in range(1,10)]
    Squares = [[(3*i+k+1, 3*j+l+1) for k in range(3) for l in range(3)] for i in range(3) for j in range(3)]
    
    # Define optimization problem and decision variables
    prob = p.LpProblem("Sudoku_Problem", p.LpMinimize)
    x = p.LpVariable.dicts("x", (range(1,10), range(1,10), range(1,10)), cat='Binary')
    
    # add constraints
    # each value occurred once at each row r and column c
    for r in range(1,10):
        for c in range(1,10):
            prob += p.lpSum([x[v][r][c] for v in range(1,10)]) == 1
            
    # each row, col, square has each value exactly once
    for v in range(1,10):
        for _ in Rows:
            prob += p.lpSum([x[v][r][c] for (r, c) in _]) == 1
        for _ in Cols:
            prob += p.lpSum([x[v][r][c] for (r, c) in _]) == 1
        for _ in Squares:
            prob += p.lpSum([x[v][r][c] for (r, c) in _]) == 1
    
    # cells should also match with input data
    for (v,r,c) in input_data:
        prob += x[v][r][c] == 1
    
    # Print and create return list of the solution
    out = {}
    sol_count = 0
    while True:
        prob.solve()
        if p.LpStatus[prob.status]=='Optimal':
            outsol = []
            sol_count+=1
            print('Found a solution!')
            for r in range(1,10):
                outrow = []
                if r in [1, 4, 7]:
                    print("+-------+-------+-------+")
                for c in range(1,10):
                    for v in range(1,10):
                        if p.value(x[v][r][c]) == 1:
                            if c in [1, 4, 7]:
                                print("| ", end ="")
                            if [v,r,c] in input_data:
                                print(colored(str(v),'red') + " ", end="")
                            else:
                                print(str(v) + " ", end="")
                            outrow.append(v)
                            if c == 9:
                                print("|\n",end="")
                outsol.append(outrow)
            print("+-------+-------+-------+")
            out[sol_count] = outsol
            # Add another constraint to find other feasible solutions
            #print(p.lpSum([x[v][r][c] for v in range(1,10) for r in range(1,10) for c in range(1,10) if p.value(x[v][r][c])==1]))
            prob += p.lpSum([x[v][r][c] for v in range(1,10) for r in range(1,10) for c in range(1,10) if p.value(x[v][r][c])==1]) <=80

        else:
            if (sol_count==0):
                print('No solution!')
            else:
                print('All solutions found')
                break

    return out,x
