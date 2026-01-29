import numpy as np

class Connect4:

    def __init__(self):
        self.W = 7
        self.H = 6
        self.board = np.zeros((self.H, self.W))
        self.all_moves = np.arange(self.W)


    def reset(self):
        self.board = np.zeros((self.H, self.W))
        return self.board
        
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(int)

    def step(self, board, move, player):
        board = board.copy()
        bcol = move
        brow = self.H-1
        # row, col = move//self.W, move%self.W
        # brow, bcol = self.H-1, col
        while True:
            if board[brow, bcol] == 0:
                board[brow, bcol] = player
                break
            brow -= 1
        done, result = self.is_terminal(board)
        return board, result, done

    def is_terminal(self, board,H = 6, W = 7):
        for r in range(H):
            for c in range(W - 3):
                val = board[r, c]
                if val in [1, -1] and val == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                    return True, val
        for r in range(H - 3):
            for c in range(W):
                val = board[r, c]
                if val in [1, -1] and val == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                    return True, val
        for r in range(H - 3):
            for c in range(W - 3):
                val = board[r, c]
                if val in [1, -1] and val == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                    return True, val
        for r in range(3, H):
            for c in range(W - 3):
                val = board[r, c]
                if val in [1, -1] and val == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                    return True, val
        if not (board == 0).any():
            return True, 0
        return False, 0
        
    def stackedStates(self, state, current_player):
        
        return np.stack((
            state == current_player, 
            state == -current_player,
            state == 0
        )).astype(np.float32)

    
    def show(self, state):
        print(state)