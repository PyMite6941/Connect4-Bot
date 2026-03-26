class Log:
    def __init__(self,filename='log.txt'):
        self.filename = filename
    
    def write(self,board):
        with open(self.filename,'a') as file:
            for i in range(0, 42, 7):
                file.write(" ".join(str(cell)[0] for cell in board[i:i+7])+"\n")
            file.write('\n')
            file.write('-'*20+'\n')
            file.write('\n')