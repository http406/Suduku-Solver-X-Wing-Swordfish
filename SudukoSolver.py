import random
import math

def Random(times, min, max, div):
    res = []
    for i in range(times):
        res.append(random.randint(min, max) / div)
    return res

def train_model():
    learning_rate = 0.1
    epochs = 5000
    weights = [*Random(2, -1000, 1000, 1000)]
    bias = [*Random(1, -10, 10, 10)]
    
    print("Training neural network...")
    for epoch in range(epochs):
        inputs = [*Random(2, 0, 400, 100)]
        if inputs[0] > inputs[1]:
            target_output = 1
        elif inputs[0] < inputs[1]:
            target_output = 0
        
        weighted_sum = (inputs[0] * weights[0] + inputs[1] * weights[1] + bias[0]) / 10
        
        if weighted_sum >= 0:
            sigmoid_activation = 1 / (1 + math.exp(-weighted_sum))
        else:
            sigmoid_activation = math.exp(weighted_sum) / (1 + math.exp(weighted_sum))

        if sigmoid_activation >= .5:
            binary_classification = 1
        else:
            binary_classification = 0

        error = target_output - sigmoid_activation

        sigmoid_derivative = sigmoid_activation * (1 - sigmoid_activation)

        gradient_weights = []
        for i in range(len(weights)): 
            gradient_weights.append(error * inputs[i] * sigmoid_derivative)

        gradient_bias = error * sigmoid_derivative

        for i in range(len(weights)):
            weights[i] += learning_rate * gradient_weights[i]
            bias[0] += learning_rate * gradient_bias
    
    print(f"The model's updated weights: {weights[0]}, {weights[1]} and bias: {bias[0]}")
    return weights, bias

def evaluate_partial_solution(partial_solution, weights, bias):
    # Convert the partial_solution to suitable input format for the neural network
    inputs = [partial_solution[i][j] for i in range(9) for j in range(9)]
    # Apply the neural network weights and bias
    weighted_sum = sum([inputs[i] * weights[i] for i in range(len(weights))]) + bias[0]
    sigmoid_activation = 1 / (1 + math.exp(-weighted_sum))
    return sigmoid_activation

def solve_sudoku(board):
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True  # Puzzle solved
    else:
        row, col = empty_cell

    for num in range(1, 10):
        if is_valid_move(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # Backtrack

    return False  # No valid number found

def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid_move(board, row, col, num):
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[i][col] for i in range(9)]:
        return False

    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False

    return True

def print_board(board):
    for row in board:
        print(row)

# Advanced Sudoku solving techniques

def forward_checking(board, row, col, num):
    # Update possible values of peers
    for i in range(9):
        if board[row][i] == 0 and num in board[row][:i] + board[row][i + 1:]:
            return False
        if board[i][col] == 0 and num in [board[j][col] for j in range(9) if j != row]:
            return False

    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == 0 and num in [board[m][n] for m in range(start_row, start_row + 3) for n in range(start_col, start_col + 3) if (m, n) != (row, col)]:
                return False

    return True

def hidden_singles(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                possible_values = set(range(1, 10))
                for k in range(9):
                    if board[i][k] != 0:
                        possible_values.discard(board[i][k])
                    if board[k][j] != 0:
                        possible_values.discard(board[k][j])
                start_row, start_col = 3 * (i // 3), 3 * (j // 3)
                for m in range(start_row, start_row + 3):
                    for n in range(start_col, start_col + 3):
                        if board[m][n] != 0:
                            possible_values.discard(board[m][n])
                if len(possible_values) == 1:
                    board[i][j] = possible_values.pop()

def naked_pairs(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                possible_values = set(range(1, 10))
                for k in range(9):
                    if board[i][k] != 0:
                        possible_values.discard(board[i][k])
                    if board[k][j] != 0:
                        possible_values.discard(board[k][j])
                start_row, start_col = 3 * (i // 3), 3 * (j // 3)
                for m in range(start_row, start_row + 3):
                    for n in range(start_col, start_col + 3):
                        if board[m][n] != 0:
                            possible_values.discard(board[m][n])
                if len(possible_values) == 2:
                    for k in range(9):
                        if k != j and board[i][k] == 0:
                            for value in possible_values:
                                if value in [board[i][k] for k in range(9)]:
                                    board[i][j] = value
                                    board[i][k] = value
                                    break
                            if board[i][j] != 0:
                                break
                        if k != i and board[k][j] == 0:
                            for value in possible_values:
                                if value in [board[k][j] for k in range(9)]:
                                    board[i][j] = value
                                    board[k][j] = value
                                    break
                            if board[i][j] != 0:
                                break
                    if board[i][j] != 0:
                        return

# Example Sudoku puzzle (0 represents empty cells)
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Train the neural network
weights, bias = train_model()

# Solve Sudoku
if solve_sudoku(board):
    print("Sudoku puzzle solved:")
    print_board(board)
    # Evaluate the solution
    evaluation = evaluate_partial_solution(board, weights, bias)
    print(f"The evaluation of the partial solution is: {evaluation}")
else:
    print("No solution exists for the Sudoku puzzle.")

