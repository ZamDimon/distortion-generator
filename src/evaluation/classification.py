"""
Class containing helper entities for classification evaluation.
"""

from typing import List

import pandas as pd
import seaborn as sn
from tabulate import tabulate

import numpy as np
from matplotlib import pyplot

from src.evaluation.authentication_system import DebugAuthenticationSystem
from src.evaluation.pair_picker import PairValidationGenerator 

from rich.progress import Progress

class ClassificationResult:
    def __init__(self, true_positive: int, true_negative: int, false_positive: int, false_negative: int) -> None:
        self._true_positive = true_positive
        self._true_negative = true_negative
        self._false_positive = false_positive
        self._false_negative = false_negative
        
    def precision(self) -> float:
        return self._true_positive / (self._true_positive + self._false_positive)
    
    def recall(self) -> float:
        return self._true_positive / (self._true_positive + self._false_negative)
    
    def f1_score(self) -> float:
        return 2*self._true_positive / (2*self._true_positive + self._false_positive + self._false_negative)
    
    def fpr(self) -> float:
        return self._false_positive / (self._false_positive + self._true_negative)
    
    def tpr(self) -> float:
        return self.recall()   
    
    def draw_summary(self) -> None:
        # Plotting confusion matrix
        confusion_matrix = [
            [self._true_negative, self._false_positive],
            [self._false_negative, self._true_positive]
        ]

        df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) 
        pyplot.show()
        
        # Printing results
        table = [
            ['Precision', self.precision()],
            ['Recall', self.recall()],
            ['F1 Score', self.f1_score()]
        ]
        print(tabulate(table, headers=['Metrics', 'Score'], tablefmt="grid"))
    
class ClassificationStatistics:
    def __init__(self, results: List[ClassificationResult] = []) -> None:
        self._results = results
        
    def add_result(self, result: ClassificationResult) -> None:
        print(f'inserted tp={result._true_positive}')
        self._results.append(result)
        print(self._results[-1]._true_positive)
        
    def pick_best_f1(self) -> ClassificationResult:
        results_sorted = sorted(self._results, key = lambda x: x.f1_score())
        return results_sorted[-1]
    
    def get_roc_curve_points(self, n_split: int = 300):
        curve_points = []
        for i in range(n_split):
            # Defining the interval
            left_interval = i / n_split
            right_interval = (i + 1) / n_split
            
            # Finding all y's for the given interval
            y_interval = [result.tpr() for result in self._results if left_interval <= result.fpr() <= right_interval]
            if len(y_interval) == 0:
                continue
            
            # Adding a point on the curve
            x_point = (left_interval + right_interval) / 2
            y_point = np.max(y_interval)
            
            if len(curve_points) > 0:
                y_point = max(curve_points[-1][1], y_point)
            curve_points.append((x_point, y_point))
        
        return curve_points
    
    def display_roc_curve(self, label: str, n_split: int = 300, color: str = 'red'):
        points = self.get_roc_curve_points(n_split=n_split)
        line, = pyplot.step([x for x, _ in points], [y for _, y in points], linewidth=2.5, color=color, label=label)
        return line

def get_statistics(
    recognition_system: DebugAuthenticationSystem,
    pair_generator: PairValidationGenerator, 
    pairs_number: int = 100,
    threshold_from: float = 0.0,
    threshold_to: float = 2.0,
    threshold_split: int = 100
) -> ClassificationStatistics:
    """
    Generates a classification statistics based on the specified pair generator and recognition system
    """
    
    # Defining a statistics
    results: List[ClassificationResult] = []
    
    with Progress() as progress:
        thresholds_to_check = np.linspace(threshold_from, threshold_to, num=threshold_split)
        task = progress.add_task("[red]Generating statistics...", total=len(thresholds_to_check))
        
        for threshold in thresholds_to_check:
            recognition_system.set_threshold(threshold)

            # true positive, true negative, false positive, false negative
            tp = tn = fp = fn = 0
            for _ in range(pairs_number):
                positive, negative = pair_generator.pick()

                positive_id = recognition_system.login(positive)
                if positive_id == DebugAuthenticationSystem.LOGIN_FAILED:
                    fn += 1
                else:
                    tp += 1

                negative_id = recognition_system.login(negative)
                if negative_id == DebugAuthenticationSystem.LOGIN_FAILED:
                    tn += 1
                else:
                    fp += 1
            
            results.append(ClassificationResult(tp, tn, fp, fn))
            progress.update(task, advance=1)

    return ClassificationStatistics(results)