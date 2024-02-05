import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.grounding_umls import ground_umls
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts
from preprocess_utils.graph_with_LM import generate_adj_data_from_grounded_concepts__use_LM
from preprocess_utils.graph_umls_with_glove import generate_adj_data_from_grounded_concepts_umls__use_glove


output_paths = {
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
}

for dname in ['medqa']:
    output_paths[dname] = {
        'statement': {
            'train': f'./data/{dname}/statement/train.statement.jsonl',
            'dev':   f'./data/{dname}/statement/dev.statement.jsonl',
            'test':  f'./data/{dname}/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': f'./data/{dname}/grounded/train.grounded.jsonl',
            'dev':   f'./data/{dname}/grounded/dev.grounded.jsonl',
            'test':  f'./data/{dname}/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': f'./data/{dname}/graph/train.graph.adj.pk',
            'adj-dev':   f'./data/{dname}/graph/dev.graph.adj.pk',
            'adj-test':  f'./data/{dname}/graph/test.graph.adj.pk',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['umls'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'umls': [
            {'func': construct_graph_umls, 'args': (output_paths['umls']['csv'], output_paths['umls']['vocab'], output_paths['umls']['rel'], output_paths['umls']['graph'], True)},
        ],
        'medqa': [
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['dev'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['dev'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['test'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['test'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['train'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['test'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-train'], args.nprocs)},
        ],
    }


    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()