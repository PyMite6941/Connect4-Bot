import os
import random

class QLearning:
    def __init__(self):
        self.Q = {}
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 0.9 # Reliance on advisors

        self.default_path = os.path.join("memory","qtable.pkl")

    def get_Q(self,state,action):
        return self.Q.get((state,action),0.0)
    
    def update_Q(self,state,action,reward,next_state,possible_next_actions):
        current = self.get_Q(state,action)
        if possible_next_actions:
            future_best = max(self.get_Q(next_state,a) for a in possible_next_actions)
        else:
            future_best = 0.0
        new_Q = current+self.alpha*(reward+self.gamma*future_best-current)
        self.Q[(state,action)] = new_Q

    def best_move(self,state,possible_actions):
        return max(possible_actions,key=lambda a: self.get_Q(state,a))
    
    def choose_moves(self,state,valid_actions,training=True):
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions) # explore
        return self.best_move(state,valid_actions) # exploit
    
    def save_Q(self,path=None):
        import pickle
        os.makedirs(os.path.dirname(self.default_path),exist_ok=True)
        if path is None:
            path = self.default_path
        with open(path,"wb") as file:
            pickle.dump(self.Q,file)

    def load_Q(self,path=None):
        import pickle
        if path is None:
            path = self.default_path
        try:
            with open(path,"rb") as file:
                self.Q = pickle.load(file)
        except FileNotFoundError:
            self.Q = {}