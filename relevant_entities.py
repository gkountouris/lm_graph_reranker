import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.query_relevant_entities import find_relevant_entities


output_paths = {
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
}

for dname in ['mixed']:
    output_paths[dname] = {
        'statement': {
            'train':  f'./data/{dname}/statement/mixed.statement.jsonl',
            'dev':  f'./data/{dname}/statement/mixed_dev.statement.jsonl',
        },
        'graph': {
            'train':  f'./data/{dname}/graph/mixed.entities.json',
            'dev':  f'./data/{dname}/graph/mixed_dev.entities.json',
        },
    }

for dname in ['BioASQ']:
    output_paths[dname] = {
        'statement': {
            'testset4':  f'./data/{dname}/statement/BioASQ-task12bPhaseA-testset4.statement.jsonl',
        },
        'graph': {
            'testset4':  f'./data/{dname}/graph/BioASQ-task12bPhaseA-testset4.entities.json',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['BioASQ'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'mixed' :[
        #     {'func': find_relevant_entities, 'args': (output_paths['mixed']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['train'], args.nprocs)},
        #     {'func': find_relevant_entities, 'args': (output_paths['mixed']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['dev'], args.nprocs)},
        # ],
        'BioASQ' :[
            {'func': find_relevant_entities, 'args': (output_paths['BioASQ']['statement']['testset4'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['testset4'], args.nprocs)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()