class Game:
    def __init__(self,rows=6,cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.last_move = None
        self.winner = None

    def possible_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]
    
    def make_move(self,col,player):
        for r in range(self.rows-1,-1,-1):
            if self.board[r][col] == 0:
                self.board[r][col] = player
                self.last_move = (r,col)
                return True
        return False
    
    def is_full(self):
        return len(self.possible_moves()) == 0
    
    def check_win(self,player):
        for r in range(self.rows):
            for c in range(self.cols):
                if all(self.board[r][c+i] == player for i in range(4) if c+i < self.cols) or all(self.board[r+i][c] == player for i in range(4) if r+i < self.rows) or all(self.board[r+i][c+i] == player for i in range(4) if r+i < self.rows and c+i < self.cols) or all(self.board[r-i][c-i] == player for i in range(4) if r-i >= 0 and c-i >= 0):
                    return {'success': True}
        return {'success': False}

    def get_state(self):
        return tuple(self.board[r][c] for r in range(self.rows) for c in range(self.cols))
    
    def is_terminal(self,player):
        result = self.check_win(player)
        if result['success']:
            self.winner = f'{player}'
            return {'end': True, 'winner': f'{player}'}
        elif self.is_full():
            self.winner = None
            return {'end': True, 'result': 'Draw'}
        else:
            return {'end': False}