import pydotplus as pdt
from collections import deque


class get_tree_pdf:
    
    def __init__(self, filename='TREE.pdf', root=None):
        dotdata = '''digraph Tree{
                node [shape = circle] ;'''
        que = deque()
        r = root
        que.append(r)
        count = 0
        if r.vis_no == -1:
            r.vis_no = count

        dotdata = dotdata + "\n{} [label=\"Feature to split on : X[{}]\\nOutput class is : {}\"];".format(count, r.feature, r.output)
        
        while len(que) != 0:
            node = que.popleft()
            for i in node.children:
                count += 1
                if node.children[i].vis_no == -1:
                    node.children[i].vis_no = count
                
                dotdata = dotdata + "\n{} [label=\"Feature to split on : X[{}]\\nOutput class is : {}\" ];".format(node.children[i].vis_no,node.children[i].feature,node.children[i].output)
                dotdata = dotdata + "\n{} -> {} [ headlabel = \"Feature Value = {}\"];".format(node.vis_no,node.children[i].vis_no,i)
                que.append(node.children[i])

        dotdata = dotdata + "\n}"
        graph = pdt.graph_from_dot_data(dotdata)
        graph.write_pdf(filename)
        print("File successfully created!")
