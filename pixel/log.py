class Log:
    def __init__(self,filename='log.txt',directory='logs/'):
        self.filename = filename
        self.directory = directory
        self.filepath = self.directory + self.filename
    
    def write(self,board):
        with open(self.filepath,'a') as file:
            for row in board:
                file.write(f'{row}\n')
            file.write('\n')
            file.write('-'*20+'\n')
            file.write('\n')