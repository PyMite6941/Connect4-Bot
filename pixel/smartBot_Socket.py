import DFS

class Socket:
    def __init__(self,game,p1,p2):
        self.game = game
        self.p1 = p1
        self.p2 = p2

    def game_to_dfs_state(self):
        flat = []
        characters = {self.p1: 1,self.p2: 2,0:0}
        for r in self.game.board:
            for item in r:
                flat.append(characters.get(item,0))
        return flat

    def get_dfs_move(self,depth=4):
        state = self.game_to_dfs_state()
        return DFS.compute(depth,state)