import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.graph_umls_with_glove_rerank import generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove


output_paths = {
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
}


for dname in ['pubmedqa']:
    output_paths[dname] = {
        'statement': {
            'train': f'./data/{dname}/statement/train_midle.statement.jsonl',
            'dev':   f'./data/{dname}/statement/dev_midle.statement.jsonl',
            'test':  f'./data/{dname}/statement/test_midle.statement.jsonl',
        },
        'grounded': {
            'train': f'./data/{dname}/grounded/train_midle.grounded.jsonl',
            'dev':   f'./data/{dname}/grounded/dev_midle.grounded.jsonl',
            'test':  f'./data/{dname}/grounded/test_midle.grounded.jsonl',
        },
        'graph': {
            'adj-train': f'./data/{dname}/graph/train_midle.graph.adj.pk',
            'adj-dev':   f'./data/{dname}/graph/dev_midle.graph.adj.pk',
            'adj-test':  f'./data/{dname}/graph/test_midle.graph.adj.pk',
        },
        'tensors': {
            'tensors-train': f'./data/{dname}/tensors/train_midle.saved_tensors.pt',
            'tensors-dev':   f'./data/{dname}/tensors/dev_midle.saved_tensors.pt',
            'tensors-test':  f'./data/{dname}/tensors/test_midle.saved_tensors.pt',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['pubmedqa'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'pubmedqa': [
            {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-dev'], output_paths['pubmedqa']['tensors']['tensors-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['test'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-test'], output_paths['pubmedqa']['tensors']['tensors-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-train'], output_paths['pubmedqa']['tensors']['tensors-train'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()