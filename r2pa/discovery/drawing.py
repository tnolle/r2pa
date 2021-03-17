import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.dag import dag_longest_path_length

from april.fs import EVALUATION_DIR
from r2pa.discovery.coder import EncodingDecodingAttributes
from april.processmining import EventLog


def draw_likelihood_graph(graph, file_name=None, coder_attributes=None):
    fig = plt.figure(1, figsize=(20, 50))

    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    edge_labels = nx.get_edge_attributes(graph, 'probability')
    node_labels = nx.get_node_attributes(graph, 'value')

    # add display names if given encoder and decoder
    if coder_attributes is not None:
        graph_add_display_names(graph, coder_attributes)
        node_labels = nx.get_node_attributes(graph, 'display_name')

    # add counts of cache to labels
    #nl2 = nx.get_node_attributes(graph, 'count_uncached')
    #for n, c in nl2.items():
    #    node_labels[n] = node_labels[n] + " " + str(c)

    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    if file_name is not None:
        fig.savefig(str(EVALUATION_DIR / file_name))
    else:
        plt.show()

    plt.close()


def draw_manual_likelihood_graph(graph, file_name=None):
    fig = plt.figure(1, figsize=(20, 50))

    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    edge_labels = nx.get_edge_attributes(graph, 'probability')
    node_labels = nx.get_node_attributes(graph, 'value')

    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    if file_name is not None:
        fig.savefig(str(EVALUATION_DIR / file_name))
    else:
        plt.show()

    plt.close()


def determine_figure_size(graph):
    length = dag_longest_path_length(graph)

    root = [n for n, d in graph.in_degree() if d == 0][0]
    graph_width = max(len(nx.descendants_at_distance(graph, root, d)) for d in range(length))

    width = graph_width * 3
    height = length * 1.5

    return width, height


def draw_and_store_likelihood_graph_with_colors(graph, dataset, file_name=None, coder_attributes=None):
    width, height = determine_figure_size(graph)
    fig = plt.figure(1, figsize=(width, height))

    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    edge_labels = nx.get_edge_attributes(graph, 'probability')
    node_labels = nx.get_node_attributes(graph, 'label')

    color_map = []
    # add display names if given encoder and decoder
    if coder_attributes is not None:
        graph_add_display_names(graph, coder_attributes)
        node_labels = nx.get_node_attributes(graph, 'display_name')

    # store graph
    nx.write_gpickle(graph, f"{str(EVALUATION_DIR / file_name)}.gpickle")

    from april.utils import microsoft_colors

    attribute_colors = microsoft_colors[3:]
    colors = dict(zip(range(dataset.num_attributes-1), attribute_colors))

    for node in graph:
        if graph.nodes[node]['display_name'] in [str(EventLog.start_symbol), str(EventLog.end_symbol)]:
            color_map.append(microsoft_colors[0])
        elif graph.nodes[node]['attribute'] == 0:
            color_map.append(microsoft_colors[2])
        else:
            color_map.append(colors[graph.nodes[node]['attribute']-1])

    # add counts of cache to labels
    # nl2 = nx.get_node_attributes(graph, 'count_uncached')
    # for n, c in nl2.items():
    #    node_labels[n] = node_labels[n] + " " + str(c)

    nx.draw_networkx_nodes(graph, pos, node_color=color_map)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    if file_name is not None:
        fig.savefig(f"{str(EVALUATION_DIR / file_name)}.pdf")
    else:
        plt.show()

    plt.close()


def graph_add_display_names(graph, coder_attributes):
    """ Add display names to a graph.
        :param graph: The graph for which the display names are to be added.
        :param coder_attributes: Decoding labels to display name. """
    # add identifier attribute
    for node in graph.nodes:
        node_attributes = graph.nodes[node]
        identifier = coder_attributes.decode(node_attributes['label'], node_attributes['attribute'])
        node_attributes['display_name'] = identifier


def add_uncached_counts_to_graph(graph, uncached_walks, number_attributes):
    encoder_decoder_attributes = EncodingDecodingAttributes.from_graph(graph, number_attributes)
    for walk in uncached_walks:
        walk = walk.split('->')[1:]  # remove double start symbol
        current_node = encoder_decoder_attributes.encode(EventLog.start_symbol, 0)
        for i in range(len(walk) - 1):
            successors = graph.successors(current_node)
            for s in successors:
                node_attributes = graph.nodes[s]
                # follow path of walk
                if node_attributes['label'] == walk[i + 1]:
                    node_attributes['count_uncached'] = node_attributes['count_uncached'] + 1
                    current_node = s
