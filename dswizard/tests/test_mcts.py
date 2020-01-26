import random
from unittest import TestCase
import networkx as nx

from dswizard.core.model import StatusType, Result
from dswizard.optimizers.structure_generators.mcts import MCTS


class TestOneHotEncoder(TestCase):

    def test_generic(self):
        search = MCTS(dataset_properties={'target_type': 'classification'})
        tree = search.tree

        for i in range(20):
            candidate = search.get_candidate(1)
            print(candidate.pipeline.steps)

            result = Result(StatusType.SUCCESS, None, random.uniform(0, 1))
            search.register_result(candidate, result)

        best = search.choose()
        print(best.steps)
        tree.highlight_path(best.id)

        for id in tree.G.nodes:
            node = tree.G.nodes[id]
            node['label'] = '{}\nn={}\nq={}'.format(node['value'].label, node['value'].visits,
                                                    node['value'].reward / node['value'].visits)

        H = nx.nx_agraph.to_agraph(search.tree.G)
        H.draw('mcts.png', prog='dot')
