import copy


class Procedure:

    def __init__(self, start_node):
        self.current_node = start_node
        self.node_sequence = [start_node]
        self.likelihood_sequence = []

    def extend(self, node, likelihood):
        self.current_node = node
        self.node_sequence.append(node)
        self.likelihood_sequence.append(likelihood)

    def length(self):
        return len(self.node_sequence)

    def tuple(self):
        return tuple(self.node_sequence), tuple(self.likelihood_sequence)


class AutomatableProcedures:

    def __init__(self, graph, minimum_sequence_length, minimum_edge_value):
        self.graph = graph
        self.minimum_sequence_length = minimum_sequence_length
        self.minimum_edge_value = minimum_edge_value

        self.root = [n for n, d in graph.in_degree() if d == 0][0]
        self.end_node = [n for n, d in graph.out_degree() if d == 0][0]

    def find(self):
        root = self.graph.nodes[self.root]['value']

        # (current node, sequence of nodes, sequence of likelihoods)
        queue = [Procedure(root)]
        automatable_procedures = set()

        while len(queue) > 0:
            procedure = queue.pop()
            outgoing_edges = self.graph.out_edges([procedure.current_node], data=True)
            for from_node, to_node, edge_value_dict in outgoing_edges:
                likelihood = edge_value_dict['probability']
                # extend sequence if edge value big enough
                if likelihood >= self.minimum_edge_value:
                    extended_procedure = copy.deepcopy(procedure)
                    extended_procedure.extend(to_node, likelihood)
                    # continue of current node is not the end node of the graph
                    if to_node == self.end_node:
                        automatable_procedures.add(procedure.tuple())
                    else:
                        queue.insert(0, extended_procedure)
                else:
                    # store sequence if it satisfies the minimum sequence length
                    # also need to stop if the end node is reached
                    if procedure.length() >= self.minimum_sequence_length or to_node == self.end_node:
                        automatable_procedures.add(procedure.tuple())
                    # start new sequence from this node on if not at the end node
                    if to_node != self.end_node:
                        queue.insert(0, Procedure(to_node))

        return automatable_procedures
