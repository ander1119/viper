import pandas as pd
import numpy as np

def accuracy(prediction, ground_truth):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
            possible_answers (list): List of possible answers.
            query_type (list): List of query types
        Returns:
            score (float): Score of the prediction.
        """
        from sklearn.metrics import precision_recall_fscore_support
        assert len(prediction) == len(ground_truth)
        score = 0

        prediction = [1 if p == 'yes' else 0 for p in prediction]
        ground_truth = [1 if g == 'yes' else 0 for g in ground_truth]

        accuracy = sum(1 for p, t in zip(prediction, ground_truth) if p == t) / len(ground_truth)

        f1 = precision_recall_fscore_support(ground_truth, prediction, average='binary')
        print(f1)
        score = {
            'precision': f1[0],
            'recall': f1[1],
            'f1': f1[2],
            'accuracy': accuracy
        }

        return score

if __name__ == '__main__':
    csv_file = '/mnt/ssd2/TropeVLM/ViperGPT/results/timos_bc/val/results_gpt35_5_plot_20_tropes.csv'
    # csv_file = '/mnt/ssd2/TropeVLM/ViperGPT/results/timos_bc/val/results_gpt4_5_plots_20_tropes.csv'

    rows = pd.read_csv(csv_file, sep='|')
    print(accuracy(np.array(rows['result'].to_list()), np.array(rows['answer'].to_list())))
    # print(predictions)
    # print(len(rows))
    # print(rows.iloc[0]['code'])