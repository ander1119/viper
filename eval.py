import numpy as np
import pandas as pd
import os
import argparse
import json
from sklearn.metrics import precision_recall_fscore_support

trope_2_cat = json.load(open('../tropes/map_trope_cat.json', 'r'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str, help="Path to the folder containing results")
    parser.add_argument('--result_type', type=str, default='csv', help="Type of the result file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if os.path.isfile(args.result):
        results = [args.result]
    else:
        results = [os.path.join(args.result, file) for file in os.listdir(args.result) if file.endswith(f'.{args.result_type}')]
    
    if args.result_type == 'json':
        json_results = []
        for result in results:
            json_results.extend(json.load(open(result, 'r')))
        answers = [x['answer'] for x in json_results]
        groundtruths = [x['groundtruth'] for x in json_results]
        tropes = [x['trope'] for x in json_results]
    else:
        dataframes = []
        for file in results:
            df = pd.read_csv(file, sep='|')
            dataframes.append(df)
        dataframes = pd.concat(dataframes, ignore_index=True)
        answers = list(dataframes['answer'])
        groundtruths = list(dataframes['groundtruth'])
        tropes = list(dataframes['trope'])

    category_result = {
        "Character Trait": {},
        "Role Interaction": {},
        "Situation": {},
        "Story Line": {},
        "Total": {}
    }
    for answer, groundtruth, trope in zip(answers, groundtruths, tropes):
        cat = trope_2_cat[trope]
        if trope not in category_result[cat]:
            category_result[cat][trope] = {
                'answers': [],
                'groundtruths': []
            }
        category_result[cat][trope]['answers'].append(1 if answer == 'yes' else 0)
        category_result[cat][trope]['groundtruths'].append(1 if groundtruth == 'yes' else 0)

        if trope not in category_result['Total']:
            category_result['Total'][trope] = {
                'answers': [],
                'groundtruths': []
            }
        category_result["Total"][trope]['answers'].append(1 if answer == 'yes' else 0)
        category_result["Total"][trope]['groundtruths'].append(1 if groundtruth == 'yes' else 0)

    for cat, result in category_result.items():
        gts = np.array([r['groundtruths'] for r in result.values()]).T
        preds = np.array([r['answers'] for r in result.values()]).T
        scores = precision_recall_fscore_support(gts, preds, average='micro')
        scores = {
            'precision': scores[0],
            'recall': scores[1],
            'f1': scores[2]
        }
        print('==============================================')
        print(cat)
        print(json.dumps(scores, indent=4))


if __name__ == '__main__':
    main()