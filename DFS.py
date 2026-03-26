import random
import connect4 as game
board=game.Board()
def reflect(l, columns):
    result=[]
    for i in range(0, len(l), columns):result+=l[i:i+columns][::-1]
    return tuple(result)
def compute(depth, state=...):
    global cache
    global initDepth
    tree, cache, initDepth=board.initTree(state), {}, depth
    if depth<1:return board.randomMove(tree["state"])
    temp=board.enumerateMoves(tree["state"])
    if len(temp)==1:return list(temp)[0]
    tree.update(temp)
    search(depth, tree)
    #print("\nCache:", cache)
    cache={}
    #print("\nEval:", tree["eval"])
    #print("\nBranch evals:", {i:tree[i]["eval"] for i in tree if i not in ("state", "turn", "eval")})
    return random.choice(tuple(i for i in tree if i not in ("state", "turn", "eval") if tree[i]["eval"][0]==tree["eval"][0]-1 and tree[i]["eval"][1]==tree["eval"][1])) if tree["eval"][1]==tree["turn"]%2+1 else random.choice(tuple(i for i in tree if i not in ("state", "turn", "eval") if tree[i]["eval"]==tree["eval"]))
def search(depth, node):
    mirror=reflect(node["state"], board.columns)
    if mirror in cache:return cache[mirror]
    state=tuple(node["state"])
    if state not in cache:
        if depth!=initDepth:node.update(board.enumerateMoves(node["state"], evaluation=node["eval"][0]))
        if depth>1:
            for i in node:
                if i not in ("state", "turn", "eval"):node[i]=search(depth-1, node[i])
        if len(node)>3:node["eval"]=board.eval(node, (initDepth-depth)%2==0)
        cache[state]={i:node[i] for i in ("state", "turn", "eval")}
    return cache[state]
#board.play()