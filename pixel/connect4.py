class Game:
    def __init__(self,p1,p2,rows=6,cols=7):
        self.p1 = p1
        self.p1_value = 1
        self.p2 = p2
        self.p2_value = 2
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
        state=[]
        for row in self.board:state+=[{self.p1: 1, self.p2: 2, 0: 0}[i] for i in row]
        def connect(l):
            for i in l:
                if i!=l[0]:return False
            return True
        result=0
        for i in range(len(state)-3):
            if (i<21 and connect(state[i:i+22:7]) or i%7<4 and connect(state[i:i+4]) or i<18 and i%7<4 and connect(state[i:i+25:8]) or i<21 and i%7>2 and connect(state[i:i+19:6])) and state[i]!=0:
                result=state[i]
        return {"success": {self.p1: 1, self.p2: 2, 0: 0}[player]==result}
        if player == self.p1:
            player_value = self.p1_value
        else:
            player_value = self.p2_value
        for r in range(self.rows):
            for c in range(self.cols):
                if all(self.board[r][c+i] == player_value for i in range(4) if c+i < self.cols) or all(self.board[r+i][c] == player_value for i in range(4) if r+i < self.rows) or all(self.board[r+i][c+i] == player_value for i in range(4) if r+i < self.rows and c+i < self.cols) or all(self.board[r-i][c-i] == player_value for i in range(4) if r-i >= 0 and c-i >= 0):
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
            return {'end': True, 'winner': 'Draw'}
        else:
            return {'end': False}