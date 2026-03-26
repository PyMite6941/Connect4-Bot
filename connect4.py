import __main__, random#, pygame
#from pygame.locals import *
def connect(l):
    for i in l:
        if i!=l[0]:return False
    return True
class Board:
    def __init__(self, *, rows=6, columns=7, connect=4):
        self.rows, self.columns, self.connect=rows, columns, connect
        self.initBoard()
        self.turn=len(self.state)-self.state.count(0)
    def initBoard(self):self.state=(0,)*self.rows*self.columns
    def initTree(self, state=...):
        if state==...:state=self.state
        return {"state":state, "turn":len(state)-state.count(0), "eval":0}
    def enumerateMoves(self, state, reasonable=True, *, evaluation=0):
        turn=len(state)-state.count(0)
        if evaluation!=0 or self.winDetection(state)!=0:return {}
        if not reasonable:return {i:{"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":(0, 0)} for i in range(self.columns) if state[i]==0}
        if turn==0 and self.rows==6 and self.columns==7 and self.connect==4:return {3:{"state":self.applyMove(state, 0, 3), "turn":1, "eval":(40, 1)}}
        result, options={}, list(range(self.columns))
        for i in range(self.columns):
            if state[i]!=0:
                options.remove(i)
                continue
        for i in options:
            temp=self.applyMove(state, turn, i)
            if self.winDetection(temp)==turn%2+1:
                result[i]={"state":temp, "turn":turn+1, "eval":(0, turn%2+1)}
                break
        if len(result)==0:
            for i in options:
                if self.winDetection(self.applyMove(state, turn+1, i))==(turn+1)%2+1:
                    result[i]={"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":(0, 0)}
                    break
        if len(result)==0:
            for i in tuple(options):
                temp=self.applyMove(state, turn, i)
                if temp[i]==0 and self.winDetection(self.applyMove(temp, turn+1, i))==(turn+1)%2+1:
                    options.remove(i)
                    continue
            for i in options:
                temp, count=self.applyMove(state, turn, i), 0
                for j in range(max(i-self.connect+1, 0), min(i+self.connect, self.columns)):
                    if temp[j]==0 and self.winDetection(self.applyMove(temp, turn, j))==turn%2+1:
                        temp2=self.applyMove(temp, turn+1, j)
                        if temp2[j]==0 and self.winDetection(self.applyMove(temp2, turn, j))==turn%2+1:
                            result[i]={"state":temp, "turn":turn+1, "eval":(1, turn%2+1)}
                            break
                        count+=1
                    if count==2:
                        result[i]={"state":temp, "turn":turn+1, "eval":(1, turn%2+1)}
                        break
                if len(result)>0:break
        if len(result)==0:result={i:{"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":(0, 0)} for i in (options if len(options)>0 else range(self.columns)) if state[i]==0}
        return result
    def randomMove(self, state):return random.choice(tuple(self.enumerateMoves(state, False)))
    def applyMove(self, state, turn, move):
        state=list(state)
        state[max(i for i in range(move, len(state), self.columns) if state[i]==0)]=turn%2+1
        return tuple(state)
    def winDetection(self, state):
        for i in range(len(state)-3):
            if (i<self.columns*(self.rows-self.connect+1) and connect(state[i:i+self.columns*self.connect:self.columns]) or i%self.columns<=self.columns-self.connect and connect(state[i:i+self.connect]) or i<self.columns*(self.rows-self.connect+1) and i%self.columns<=self.columns-self.connect and connect(state[i:i+(self.columns+1)*self.connect:self.columns+1]) or i<self.columns*(self.rows-self.connect+1) and i%self.columns>=self.connect-1 and connect(state[i:i+(self.columns-1)*self.connect:self.columns-1])) and state[i]!=0:return state[i]
        return 0 if 0 in state else -1
    def eval(self, branch, ownTurn):
        evals, temp=tuple(branch[i]["eval"][0] for i in branch if i not in ("state", "turn", "eval")), tuple(branch[i]["eval"][1] for i in branch if i not in ("state", "turn", "eval"))
        if ownTurn:return (min((evals[i] for i in range(len(evals)) if temp[i]==branch["turn"]%2+1))+1, branch["turn"]%2+1) if branch["turn"]%2+1 in temp else ((max, min)[branch["turn"]%2](evals[i] for i in range(len(evals)) if temp[i]==0), 0) if 0 in temp else (max(evals), (branch["turn"]+1)%2+1)
        else:return (min((evals[i] for i in range(len(evals)) if temp[i]==branch["turn"]%2+1))+1, branch["turn"]%2+1) if branch["turn"]%2+1 in temp else (sum((0, 1, -1)[temp[i]] if temp[i]!=0 else evals[i] for i in range(len(evals)))/len(evals), 0) if 0 in temp else (max(evals), (branch["turn"]+1)%2+1)
    def display(self, window, humanPlayer=1):
        window.fill((0, 0, 0))
        if status=="difficulty":
            window.blit(pygame.font.Font(None, 90).render("Difficulty (0 - 10): "+textInput, True, (255, 255, 255)), (40, 275))
            if invalidInput:window.blit(pygame.font.Font(None, 30).render("Please input an integer between 0 and 10, inclusive.", True, (255, 0, 0)), (45, 350))
        elif status=="player":
            window.blit(pygame.font.Font(None, 100).render("Player: "+textInput, True, (255, 255, 255)), (40, 275))
            if invalidInput:window.blit(pygame.font.Font(None, 30).render("Please input either 1 or 2.", True, (255, 0, 0)), (45, 350))
        else:
            temp=self.winDetection(self.state)
            window.blit(pygame.font.Font(None, 30).render(("Click on a column to play in that column." if humanPlayer!=0 else "") if temp==0 else "It's a tie." if temp==-1 else f"{'Player' if humanPlayer!=0 else 'Computer'} {temp} wins!" if __name__=="__main__" or humanPlayer==0 else "The player wins!" if temp==humanPlayer else "The computer wins!", True, (255, 255, 255)), (40, 25))
            if __name__!="__main__" and humanPlayer!=0:
                window.blit(pygame.font.Font(None, 30).render("You are:", True, (255, 255, 255)), (500, 25))
                pygame.draw.circle(window, (255, 255 if humanPlayer==1 else 0, 0), (610, 35), 20)
            for i in range(len(self.state)):pygame.draw.circle(window, (127, 127, 127) if self.state[i]==0 else (255, 255 if self.state[i]==1 else 0, 0), ((i%self.columns*2+1)*350/self.columns, (i//self.columns*2+1)*300/self.rows+50), 37.5*min(7/self.columns, 6/self.rows))
        pygame.display.update()
    def play(self):
        pygame.init()
        self.initBoard()
        global screen, textInput, status, invalidInput
        screen, textInput, status, invalidInput, humanPlayer, clock, time=pygame.display.set_mode((700, 650)), "", "difficulty" if __name__!="__main__" else "playing", False, -1, pygame.time.Clock(), 0
        print()
        self.display(screen, humanPlayer)
        while True:
            move=-1
            for event in pygame.event.get():
                if event.type==QUIT or event.type==KEYUP and event.key==K_ESCAPE:
                    pygame.quit()
                    return
                elif event.type==MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0] and status=="playing" and (__name__=="__main__" or self.turn%2+1==humanPlayer) and abs(event.pos[0]%(700/self.columns)-350/self.columns)<=30*min(7/self.columns, 6/self.rows) and event.pos[1]>=50:move=event.pos[0]*self.columns//700
                elif event.type==KEYDOWN:
                    if event.key==K_RETURN:
                        if textInput=="" or status=="difficulty" and int(textInput) not in range(11) or status=="player" and int(textInput) not in range(3):invalidInput=True
                        elif status=="difficulty":self.difficulty, status, textInput=int(textInput), "player", ""
                        elif status=="player":
                            humanPlayer, status=int(textInput), "playing"
                            self.display(screen, humanPlayer)
                    elif event.key==K_BACKSPACE:textInput, invalidInput=textInput[:-1], False
                    elif pygame.key.name(event.key).isnumeric():
                        if invalidInput:textInput=""
                        textInput+=pygame.key.name(event.key)
                        invalidInput=False
            if status=="playing" and self.winDetection(self.state)==0:
                if __name__!="__main__" and self.turn%2+1!=humanPlayer:
                    slots=self.state[:7].count(0)
                    clock.tick()
                    move=__main__.compute(self.difficulty if self.difficulty<6 or self.turn<15 or self.rows!=6 or self.columns!=7 or self.connect!=4 else self.difficulty+1 if self.turn<25 else self.difficulty+3 if slots>5 else self.difficulty+6 if slots==5 else 17)
                    clock.tick()
                    temp=clock.get_time()
                    time+=temp
                    print("\nComputer played:", move)
                    print("\nTime taken:", temp/1000)
                    print()
                if move in self.enumerateMoves(self.state, False):
                    self.state=self.applyMove(self.state, self.turn, move)
                    self.turn+=1
                    if __name__!="__main__" and self.winDetection(self.state)!=0:print("\nAverage time taken:", time/(1000*(self.turn if humanPlayer==0 else self.turn//2 if humanPlayer==1 else (self.turn+1)//2)), "\n")
            self.display(screen, humanPlayer)
#if __name__=="__main__":Board().play()