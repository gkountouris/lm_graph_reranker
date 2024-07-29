import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.graph_umls_with_glove import generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove


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
        # 'grounded': {
            # 'train': f'./data/{dname}/grounded/train_midle.grounded.jsonl',
            # 'dev':   f'./data/{dname}/grounded/dev_midle.grounded.jsonl',
            # 'test':  f'./data/{dname}/grounded/test_midle.grounded.jsonl',
        # },
        'graph': {
            'adj-train': f'./data/{dname}/graph/train_midle_bm25.graph.adj.pk',
            'adj-dev':   f'./data/{dname}/graph/dev_midle_bm25.graph.adj.pk',
            'adj-test':  f'./data/{dname}/graph/test_midle_bm25.graph.adj.pk',
        },
        'tensors': {
            'tensors-train': f'./data/{dname}/tensors/train_midle_bm25.saved_tensors.pt',
            'tensors-dev':   f'./data/{dname}/tensors/dev_midle_bm25.saved_tensors.pt',
            'tensors-test':  f'./data/{dname}/tensors/test_midle_bm25.saved_tensors.pt',
        },
    }

for dname in ['BioASQ']:
    output_paths[dname] = {
        'statement': {
            # 'train':  f'./data/{dname}/statement/training11b.statement.jsonl',
            # 'dev':    f'./data/{dname}/statement/dev11b.statement.jsonl',
            # 'test1':  f'./data/{dname}/statement/11B1_golden.statement.jsonl',
            # 'test2':  f'./data/{dname}/statement/11B2_golden.statement.jsonl',
            # 'test3':  f'./data/{dname}/statement/11B3_golden.statement.jsonl',
            # 'test4':  f'./data/{dname}/statement/11B4_golden.statement.jsonl', 
            # 'challenge1':  f'./data/{dname}/statement/BioASQ-task12bPhaseA-testset3.statement.jsonl',
            # 'challenge2':  f'./data/{dname}/statement/BioASQ-task12bPhaseA-testset4.statement.jsonl',
            'bioasq8':  f'./data/{dname}/statement/8B1_golden.statement.jsonl',
        },
        'graph': {
            # 'adj-train':  f'./data/{dname}/graph/training11b_bm25.graph.adj.pk',
            # 'adj-dev':    f'./data/{dname}/graph/dev11b_bm25.graph.adj.pk',
            # 'adj-test1':  f'./data/{dname}/graph/11B1_golden_bm25.graph.adj.pk',
            # 'adj-test2':  f'./data/{dname}/graph/11B2_golden_bm25.graph.adj.pk',
            # 'adj-test3':  f'./data/{dname}/graph/11B3_golden_bm25.graph.adj.pk',
            # 'adj-test4':  f'./data/{dname}/graph/11B4_golden_bm25.graph.adj.pk',
            # 'adj-challenge1':  f'./data/{dname}/graph/BioASQ-task12bPhaseA-testset3.graph.adj.pk',
            # 'adj-challenge2':  f'./data/{dname}/graph/BioASQ-task12bPhaseA-testset4.graph.adj.pk',
            'adj-bioasq8':  f'./data/{dname}/graph/8B1_golden.graph.adj.pk',
        },
        'tensors': {
            # 'tensors-train':  f'./data/{dname}/tensors/training11b_bm25.saved_tensors.pt',
            # 'tensors-dev':    f'./data/{dname}/tensors/dev11b_bm25.saved_tensors.pt',
            # 'tensors-test1':  f'./data/{dname}/tensors/11B1_golden_bm25.saved_tensors.pt',
            # 'tensors-test2':  f'./data/{dname}/tensors/11B2_golden_bm25.saved_tensors.pt',
            # 'tensors-test3':  f'./data/{dname}/tensors/11B3_golden_bm25.saved_tensors.pt',
            # 'tensors-test4':  f'./data/{dname}/tensors/11B4_golden_bm25.saved_tensors.pt',
            # 'tensors-challenge1':  f'./data/{dname}/tensors/BioASQ-task12bPhaseA-testset3.saved_tensors.pt',
            # 'tensors-challenge2':  f'./data/{dname}/tensors/BioASQ-task12bPhaseA-testset4.saved_tensors.pt',
            'tensors-bioasq8':  f'./data/{dname}/tensors/8B1_golden.saved_tensors.pt',
        },
    }


for dname in ['mixed']:
    output_paths[dname] = {
        'statement': {
            'mini':  f'./data/{dname}/statement/mini_100.statement.jsonl',
            'train':  f'./data/{dname}/statement/mixed.statement.jsonl',
            'dev':  f'./data/{dname}/statement/mixed_dev.statement.jsonl',
        },
        'graph': {
            'adj-mini':  f'./data/{dname}/graph/mini_100.graph.adj.pk',
            'adj-train':  f'./data/{dname}/graph/mixed.graph.adj.pk',
            'adj-dev':  f'./data/{dname}/graph/mixed_dev.graph.adj.pk',
        },
        'tensors': {
            'tensors-mini':  f'./data/{dname}/tensors/mini_100.saved_tensors.pt',
            'tensors-train':  f'./data/{dname}/tensors/mixed.saved_tensors.pt',
            'tensors-dev':  f'./data/{dname}/tensors/mixed_dev.saved_tensors.pt',
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['mixed'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'pubmedqa': [
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-dev'], output_paths['pubmedqa']['tensors']['tensors-dev'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['test'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-test'], output_paths['pubmedqa']['tensors']['tensors-test'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['pubmedqa']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['pubmedqa']['graph']['adj-train'], output_paths['pubmedqa']['tensors']['tensors-train'], args.nprocs)},
        # ],
        # 'BioASQ': [
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['test1'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test1'], output_paths['BioASQ']['tensors']['tensors-test1'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['test2'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test2'], output_paths['BioASQ']['tensors']['tensors-test2'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['test3'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test3'], output_paths['BioASQ']['tensors']['tensors-test3'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['test4'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test4'], output_paths['BioASQ']['tensors']['tensors-test4'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-dev'], output_paths['BioASQ']['tensors']['tensors-dev'], args.nprocs)},
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-train'], output_paths['BioASQ']['tensors']['tensors-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['challenge2'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-challenge2'], output_paths['BioASQ']['tensors']['tensors-challenge2'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['BioASQ']['statement']['bioasq8'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-bioasq8'], output_paths['BioASQ']['tensors']['tensors-bioasq8'], args.nprocs)},
        # ],
        'mixed' :[
        #     {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['mixed']['statement']['mini'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['adj-mini'], output_paths['mixed']['tensors']['tensors-mini'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['mixed']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['adj-train'], output_paths['mixed']['tensors']['tensors-train'], args.nprocs, True)},
            {'func': generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove, 'args': (output_paths['mixed']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['adj-dev'], output_paths['mixed']['tensors']['tensors-dev'], args.nprocs, False)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()