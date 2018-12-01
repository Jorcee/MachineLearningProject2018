from sys import setrecursionlimit
setrecursionlimit(1000000) 

class Node:
    def __init__(self,item):
        self.num = 1
        self.similarity=1
        #item is index of samples for leafnodes
        # similarity of two children for other nodes
        self.index = item
        self.child1 = None
        self.child2 = None

def combine(a,b,similarity):
    c=Node(None)
    c.child1 = a
    c.child2 = b
    c.num = a.num+b.num
    c.index=min(a.index,b.index)
    c.similarity=similarity
    return c

def node_simularity(correlation,node1,node2):
    #return correlation[preorder(node1)][:,preorder(node2)].sum()
    return correlation[node1.index,node2.index]

def preorder(root):  # 先序遍历
    # if ((root is None)) :
    #     return []
    if(root.num==1):
        return [root.index]
    else:
        return preorder(root.child1)+preorder(root.child2)