class DecisionTreeNode:
    def __init__(self, feature, output):
        self.feature = feature
        self.output = output
        self.children = {}
        self.vis_no = -1
    
    def add_child(self, feature, obj):
        self.children[feature] = obj
