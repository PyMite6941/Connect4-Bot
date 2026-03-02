class Log:
    def __init__(self,filename='log.txt'):
        self.filename = filename
    
    def write(self,board):
        with open(self.filename,'a') as file:
            for row in board:
                file.write(' '.join(str(cell) for cell in row)+'\n')
            file.write('\n')
            file.write('-'*20+'\n')
            file.write('\n')    