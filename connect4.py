import __main__, random, pygame
from pygame.locals import *
def connect(l):
    for i in l:
        if i!=l[0]:return False
    return True
class Board:
    def __init__(self):
        self.state=[0]*42
        self.turn=len(self.state)-self.state.count(0)
    def initTree(self, state=None):
        if state is None:state=self.state
        return {"state":state, "turn":len(state)-state.count(0), "eval":0}
    def enumerateMoves(self, state, reasonable=True, *, evaluation=0):
        turn=len(state)-state.count(0)
        if evaluation!=0 or self.winDetection(state)!=0:return {}
        if not reasonable:return {i:{"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":0} for i in range(7) if state[i]==0}
        if turn==0:return {3:{"state":self.applyMove(state, 0, 3), "turn":1, "eval":1}}
        result, options={}, list(range(7))
        for i in range(7):
            if state[i]!=0:
                options.remove(i)
                continue
        for i in options:
            temp=self.applyMove(state, turn, i)
            if self.winDetection(temp)==turn%2+1:
                result[i]={"state":temp, "turn":turn+1, "eval":[0, 1, -1][turn%2+1]}
                break
        if len(result)==0:
            for i in options:
                if self.winDetection(self.applyMove(state, turn+1, i))==(turn+1)%2+1:
                    result[i]={"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":0}
                    break
        if len(result)==0:
            for i in list(options):
                temp=self.applyMove(state, turn, i)
                if temp[i]==0 and self.winDetection(self.applyMove(temp, turn+1, i))==(turn+1)%2+1:
                    options.remove(i)
                    continue
            for i in options:
                temp, count=self.applyMove(state, turn, i), 0
                for j in range(max(i-3, 0), min(i+4, 7)):
                    if temp[j]==0 and self.winDetection(self.applyMove(temp, turn, j))==turn%2+1:
                        temp2=self.applyMove(temp, turn+1, j)
                        if temp2[j]==0 and self.winDetection(self.applyMove(temp2, turn, j))==turn%2+1:
                            result[i]={"state":temp, "turn":turn+1, "eval":[0, 1, -1][turn%2+1]}
                            break
                        count+=1
                    if count==2:
                        result[i]={"state":temp, "turn":turn+1, "eval":[0, 1, -1][turn%2+1]}
                        break
                if len(result)>0:break
        if len(result)==0:result={i:{"state":self.applyMove(state, turn, i), "turn":turn+1, "eval":0} for i in (options if len(options)>0 else range(7)) if state[i]==0}
        return result
    def randomMove(self, state):return random.choice(list(self.enumerateMoves(state, False)))
    def applyMove(self, state, turn, move):
        state=list(state)
        state[max([i for i in range(move, len(state), 7) if state[i]==0])]=turn%2+1
        return state
    def winDetection(self, state):
        for i in range(len(state)-3):
            if (i<21 and connect(state[i:i+22:7]) or i%7<4 and connect(state[i:i+4]) or i<18 and i%7<4 and connect(state[i:i+25:8]) or i<21 and i%7>2 and connect(state[i:i+19:6])) and state[i]!=0:return state[i]
        return 0 if 0 in state else -1
    def eval(self, branch, ownTurn):
        evals, temp=[branch[i]["eval"] for i in branch if i not in ["state", "turn", "eval"]], [1, -1][branch["turn"]%2]
        if ownTurn:return [max, max, min][temp](evals)
        else:return temp if temp in evals else sum(evals)/len(evals)
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
            for i in range(len(self.state)):pygame.draw.circle(window, (127, 127, 127) if self.state[i]==0 else (255, 255 if self.state[i]==1 else 0, 0), (i%7*100+50, i//7*100+100), 35)
        pygame.display.update()
    def play(self):
        pygame.init()
        self.__init__()
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
                elif event.type==MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0] and status=="playing" and (__name__=="__main__" or self.turn%2+1==humanPlayer) and 15<=event.pos[0]%100<=85 and event.pos[1]>=50:move=event.pos[0]//100
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
                    move=__main__.compute(self.difficulty if self.difficulty<6 or self.turn<15 else self.difficulty+1 if self.turn<25 else self.difficulty+3 if slots>5 else self.difficulty+6 if slots==5 else 17)
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
if __name__=="__main__":Board().play()