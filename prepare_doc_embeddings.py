import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.graph_umls_doc_embeddings import generate_adj_data_from_grounded_concepts_umls__use_glove
from tqdm import tqdm

output_paths = {
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['pubmed_processed'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    for i in tqdm(range(237)):
        routines = {
            'pubmed_processed': [
                {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (f'data/pubmed_processed/statement/pubmed_eval_embeddings_{i:04d}.jsonl', output_paths['umls']['graph'], output_paths['umls']['vocab'], f'data/pubmed_processed/graph/pubmed_eval_embeddings_{i:04d}.graph.adj.pk', args.nprocs)},
            ],
        }

        for rt in args.run:
            for rt_dic in routines[rt]:
                rt_dic['func'](*rt_dic['args'])

        print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()

