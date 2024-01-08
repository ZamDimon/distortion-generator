from pathlib import Path

from matplotlib import pyplot

from src.evaluation.classification import ClassificationStatistics

class ROCPlotter:
    """
    Class responsible for displaying the ROC curve
    """
    
    def __init__(self) -> None:
        pass
    
    def plot_from_statistics(
        self,
        statistics_generated: ClassificationStatistics, 
        statistics_real: ClassificationStatistics,
        save_path: Path = None  
    ) -> None:
        """
        Displays the ROC curve from two statistics entities provided.
        
        Parameters:
            - statistics_generated (ClassificationStatistics) - statistics of an authentication system using generated images
            - statistics_real (ClassificationStatistics) - classical authentication system statistics without modifications
            - save_path (Path, optional) - path for saving image in. If None, image will not be saved. Defaults to None.
        """
        
        pyplot.style.use('seaborn-v0_8-white') # Choose pyplot styling
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 12}
        pyplot.rc('font', **font)
        pyplot.rc('axes', labelsize=16)
        pyplot.figure(figsize=(8, 8), dpi=300)
        pyplot.grid()
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)
        pyplot.plot([0.0, 1.0], [0.0, 1.0], linewidth=1.5, color='gray', linestyle='--')
        line_1 = statistics_generated.display_roc_curve(label='Real images', n_split=2000, color='blue')
        line_2 = statistics_real.display_roc_curve(label='Generated templates', n_split=2000, color='red')
        pyplot.legend(handles=[line_1, line_2])
        pyplot.title("Authentication System ROC Curve")
        pyplot.xlabel("FPR")
        pyplot.ylabel("TPR")
        if save_path is not None:
            pyplot.savefig(save_path)
            
        pyplot.show()
    