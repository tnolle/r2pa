import networkx as nx

from april.fs import EVALUATION_DIR
from r2pa.discovery.automatable_procedures import AutomatableProcedures


def find_automatable_procedures_from_graph(file_name, minimum_sequence_length, minimum_edge_value):
    # load graph from file
    graph = nx.read_gpickle(f"{EVALUATION_DIR / file_name}.gpickle")
    # find automatable procedures
    automatable_procedures = AutomatableProcedures(graph=graph, minimum_sequence_length=minimum_sequence_length,
                                                   minimum_edge_value=minimum_edge_value)
    procedures = automatable_procedures.find()
    # convert node identifiers to display names
    decoded_procedures = [([graph.nodes[n]['display_name'] for n in procedure], likelihoods)
                          for procedure, likelihoods in procedures]
    return decoded_procedures


if __name__ == '__main__':
    procedures = find_automatable_procedures_from_graph(file_name="log2_BINetV3", minimum_sequence_length=2,
                                                        minimum_edge_value=0.8)

    print(*procedures, sep='\n')
