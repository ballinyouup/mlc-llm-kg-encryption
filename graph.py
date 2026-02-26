class Node:
    def __init__(self, subject, predicate, obj):
        self.subject = subject
        self.predicate = predicate
        self.obj = obj

class Graph:
    def __init__(self):
        self.nodes: list[Node] = []