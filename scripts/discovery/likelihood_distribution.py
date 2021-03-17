from april import Dataset

import matplotlib.pyplot as plt
import numpy as np

from r2pa.discovery.process_model import ProcessModel

dataset_name = 'huge-0.3-1'
dataset = Dataset(dataset_name, use_event_attributes=True)
ground_truth_model = ProcessModel(dataset=dataset)
process_model = ground_truth_model.graph

edges = process_model.edges._adjdict

likelihoods = []

for node, edge_dict in edges.items():
    for out_node, likelihood_dict in edge_dict.items():
        likelihoods.extend(likelihood_dict.values())

number_bins = 50

fig, axs = plt.subplots(tight_layout=True)

axs.hist(likelihoods, cumulative=True, density=-1, bins=number_bins, range=(0, 1))

plt.xlabel("Likelihood")
plt.ylabel("Cumulative Density")

plt.xticks(ticks=np.arange(0, 1.01, 0.04), rotation=90)
plt.yticks(ticks=np.arange(0, 1.01, 0.04))

plt.grid(True)

plt.savefig(f"likelihoods_{dataset_name}.pdf", dpi=200, pad_inches='tight')

plt.show()
