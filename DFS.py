import random
import connect4 as game
board=game.Board()
def compute(depth, state=[]):
    tree, cache, initDepth=board.initTree(state), {}, depth
    def search(depth, node):
        nonlocal cache
        def reflect(l, columns):
            result=[]
            for i in range(0, len(l), columns):result+=l[i:i+columns][::-1]
            return result
        mirror=tuple(reflect(node["state"], board.columns))
        if mirror in cache:return cache[mirror]
        state=tuple(node["state"])
        if state not in cache:
            if depth!=initDepth:node.update(board.enumerateMoves(node["state"], reasonable=True, evaluation=node["eval"][0]))
            if depth>1:
                for i in node:
                    if i not in ("state", "turn", "eval"):node[i]=search(depth-1, node[i])
            else:
                for i in node:
                    if i not in ("state", "turn", "eval"):node[i]=node[i]["eval"]
            if len(node)>3:node["eval"]=board.eval(node, (initDepth-depth)%2==0)
            cache[state]=node["eval"]
        return cache[state]
    if depth<1:return board.randomMove(tree["state"])
    temp=board.enumerateMoves(tree["state"], reasonable=True)
    if len(temp)==1:return tuple(temp)[0]
    tree.update(temp)
    search(depth, tree)
    #print("\nCache:", cache)
    cache={}
    #print("\nEval:", tree["eval"])
    #print("\nMove evals:", {i:tree[i] for i in tree if i not in ("state", "turn", "eval")})
    return random.choice(tuple(i for i in tree if i not in ("state", "turn", "eval") if tree[i][0]==tree["eval"][0]-1 and tree[i][1]==tree["eval"][1])) if tree["eval"][1]==tree["turn"]%2+1 else random.choice(tuple(i for i in tree if i not in ("state", "turn", "eval") if tree[i]==tree["eval"]))
#board.play()