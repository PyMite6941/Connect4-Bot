import random
import connect4 as game
board=game.Board()
def reflect(l, columns):
    result=[]
    for i in range(0, len(l), columns):result+=l[i:i+columns][::-1]
    return result
def compute(depth, state=None):
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
    #print("\nBranch evals:", {i:tree[i]["eval"] for i in tree if i not in ["state", "turn", "eval"]})
    return random.choice([i for i in tree if i not in ["state", "turn", "eval"] if tree[i]["eval"]==tree["eval"]])
def search(depth, node):
    mirror=tuple(reflect(node["state"], 7 if game.__name__=="connect4" else 3 if game.__name__=="tictactoe" else 1))
    if mirror in cache:return cache[mirror]
    state=tuple(node["state"])
    if state not in cache:
        if depth!=initDepth:node.update(board.enumerateMoves(node["state"], evaluation=node["eval"]))
        if depth>1:
            for i in node:
                if i not in ["state", "turn", "eval"]:
                    node[i]=search(depth-1, node[i])
                    if(node[i]["eval"]==[1, -1][node["turn"]%2]):break
        if len(node)>3:node["eval"]=board.eval(node, (initDepth-depth)%2==0)
        cache[state]={i:node[i] for i in ["state", "turn", "eval"]}
    return cache[state]