"""Interface plotting
"""
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    """Class aggregates all functions for plotting.
       If a new plotting functions is required, enhance this class.
    """

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIG_SIZE = 12
    BIGGER_SIZE = 14
    EVEN_BIGGER_SIZE=16
    VERY_BIG_SIZE = 18



    @classmethod
    def plot_analysis_results_line(cls,x_data,y_data,x_label,y_label,title,path) -> None:
        """Plots the results of the analysis package as a line chart.

        Args:
            x_data (List[float]): Data for x axis
            y_data (List[float]): Data for y axis
            x_label (str): Label for x axis
            y_label (str): Label for y axis
            path (str): Path to save plot
        """
        plt.rcdefaults()
        plt.rc('legend', fontsize=cls.BIG_SIZE)
        plt.rc('axes',labelsize=cls.EVEN_BIGGER_SIZE)
        plt.rc('axes', titlesize=cls.EVEN_BIGGER_SIZE)  
        plt.rc('font', size=cls.BIG_SIZE)   
        plt.rc('xtick', labelsize=cls.BIG_SIZE)    
        plt.rc('ytick', labelsize=cls.BIG_SIZE)    

        fig, ax = plt.subplots(1,1,figsize=(9,6))
        ax.set_prop_cycle(color=['royalblue', 'darkorange', 'darkcyan'])
        for y in y_data:
            ax.plot(x_data, y,)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_ylim(0, 0.1)
        ax.legend(['Dim 10', 'Dim 50', 'Dim 100'],loc='upper left'),
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    @classmethod
    def plot_analysis_results_bar(cls,x_mc_samples,y_data,title,y_label,path) -> None:
        """Plots the results of the analysis package as a bar chart.

        Args:
            x_mc_samples (List[int]): MC Sample sizes that are investigated
            y_data (List[float]): Data for y axis
            title (str): Title of plot
            y_label (str): Label for y axis
            path (str): Path to save plot
        """
        plt.rcdefaults()
        plt.rc('legend', fontsize=cls.BIG_SIZE)
        plt.rc('axes',labelsize=cls.EVEN_BIGGER_SIZE)
        plt.rc('axes', titlesize=cls.EVEN_BIGGER_SIZE)  
        plt.rc('font', size=cls.BIG_SIZE)   
        plt.rc('xtick', labelsize=cls.EVEN_BIGGER_SIZE)    
        plt.rc('ytick', labelsize=cls.BIG_SIZE)      
 

        dimensions = ['Dim 10', 'Dim 50', 'Dim 100']

        data = {}
        for idx,y_data_dim in enumerate(y_data):
            data[dimensions[idx]] = y_data_dim


        samples = []

        for sample in x_mc_samples:
            samples.append(f"Sample Size: {sample}")

        x = np.arange(len(samples))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(1,1,figsize=(11,8),layout='constrained')

        for attribute, measurement in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1


        ax.set_prop_cycle(color=['royalblue', 'darkorange', 'darkcyan'])
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(x + width, samples)
        #ax.legend(loc='upper left', ncols=3)
        ax.legend(loc='best', ncols=3)
        #ax.set_ylim(0, 6)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    @classmethod
    def plot_model_train_line(cls,x,y,path) -> None:
        """Plots training data as a line chart.

        Args:
            x (List[int]): Data for x axis
            y (List[float]): Data for y axis
            path (str): Path to save plot
        """
        plt.rcdefaults()
        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(1,1,figsize=(9,6))
        ax.set_prop_cycle(color=['royalblue', 'darkorange', 'darkcyan'])
        ax.semilogy(x, y)
        #ax.semilogy(xrange, [e[5:8] for e in self.error_hist], linestyle='--')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('Errors')
        ax.legend(['$L^1$', '$L^2$', '$L^{\infty}$']),
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    @classmethod
    def plot_model_result_surface(cls,X,Y,Z,path) -> None:
        """Plots a three dimensional surface of the input data.

        Args:
            X (List[float]): Data for x axis
            Y (List[float]): Data for y axis
            Z (List[float]): Data for z axis
            path (str): Path to save plot
        """
        plt.rcdefaults()

        fig = plt.figure(figsize=(11,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z,cmap='viridis')
        ax.view_init(45,210)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
