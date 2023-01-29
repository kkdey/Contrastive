### Dependencies are:
# numpy - for matrix manipulation
# sklearn - for implementing spectral clustering and standard PCA
###

from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from sklearn import cluster
from sklearn.decomposition import PCA


class CPCA(object):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    """

    # Getter methods for various attributes
    def get_data(self):
        return self.data
    def get_bg(self):
        return self.bg
    def get_fg(self):
        return self.fg
    def get_active_labels(self):
        return self.active_labels
    def get_pca_directions(self):
        return self.pca_directions
    def get_active_pca_projected(self):
        return self.active_pca
    def get_affinity_matrix(self):
        return self.affinity_matrix

    # A helper method to standardize arrays
    def standardize_array(self, array):
        standardized_array =  (array-np.mean(array,axis=0)) / np.std(array,axis=0)
        return np.nan_to_num(standardized_array)

    #stores
    def __init__(self, n_components=2, standardize=True, verbose=False):
        self.standardize = standardize
        self.n_components = n_components
        self.verbose = verbose
        self.fitted = False

        """
        Finds the covariance matrices of the foreground and background datasets,
        and then transforms the foreground dataset based on the principal contrastive components

        Parameters: see self.fit() and self.transform() for parameter description

        """



    def fit_transform(self, foreground, background, plot=False, gui=False, alpha_selection='auto', n_alphas=40,  max_log_alpha=3, n_alphas_to_return=4, active_labels = None, colors=None, legend=None, alpha_value=None, return_alphas=False):
        self.fit(foreground, background)
        return self.transform(dataset=self.fg, alpha_selection=alpha_selection,  n_alphas=n_alphas, max_log_alpha=max_log_alpha, n_alphas_to_return=n_alphas_to_return, plot=plot, gui=gui, active_labels=active_labels, colors=colors, legend=legend, alpha_value=alpha_value, return_alphas=return_alphas)

        """
        Computes the covariance matrices of the foreground and background datasets

        Parameters
        -----------
        foreground: array, shape (n_data_points, n_features)
            The dataset in which the interesting directions that we would like to discover are present or enriched

        background : array, shape (n_data_points, n_features)
            The dataset in which the interesting directions that we would like to discover are absent or unenriched

        preprocess_with_pca_dim: int
            If this parameter is provided (and it is greater than n_features), then both the foreground and background
            datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
            but n_features > 1,000, a preliminary round of PCA is automatically performed to reduce the dimensionality to 1,000.
        """

    def fit(self, foreground, background, preprocess_with_pca_dim=None):
        # Housekeeping
        self.pca_directions = None
        self.bg_eig_vals = None
        self.affinity_matrix = None

        # Datasets and dataset sizes
        self.fg = foreground
        self.bg = background
        self.n_fg, self.features_d = foreground.shape
        self.n_bg, self.features_d_bg = background.shape

        if not(self.features_d==self.features_d_bg):
            raise ValueError('The dimensionality of the foreground and background datasets must be the same')

        

        #Center the background and foreground data
        self.bg = self.bg - np.mean(self.bg, axis=0)
        if self.standardize: #Standardize if specified
            self.bg = self.standardize_array(self.bg)

        self.fg = self.fg - np.mean(self.fg, axis=0)
        if self.standardize: #Standardize if specified
            self.fg = self.standardize_array(self.fg)

       
        
        if (self.verbose):
            print("Data loaded and preprocessed")

        print(self.fg.shape)
        print(self.bg.shape)
        #Calculate the covariance matrices
        self.bg_cov = self.bg.T.dot(self.bg)/(self.bg.shape[0]-1)
        self.fg_cov = self.fg.T.dot(self.fg)/(self.n_fg-1)

        # if (self.verbose):
        #     print("Covariance matrices computed")

        self.fitted = True


    def transform(self, dataset, alpha_selection='auto', n_alphas=40, max_log_alpha=3, n_alphas_to_return=4, plot=False, gui=False, active_labels = None, colors=None, legend=None, alpha_value=None, return_alphas=False):
        if (self.fitted==False):
            raise ValueError("This model has not been fit to a foreground/background dataset yet. Please run the fit() or fit_transform() functions first.")
        if not(alpha_selection=='auto' or alpha_selection=='manual' or alpha_selection=='all'):
            raise ValueError("Invalid argument for parameter alpha_selection: must be 'auto' or 'manual' or 'all'")
        if (alpha_selection=='all' and plot==True):
            raise ValueError('The plot parameter cannot be set to True if alpha_selection is set to "all"')
        if ((alpha_selection=='all' or alpha_selection=='manual') and gui==True):
            raise ValueError('The gui parameter cannot be set to True if alpha_selection is set to "all" or "manual"')
        if ((gui==True or plot==True) and not(self.n_components==2)):
            raise ValueError('The gui and plot parameters modes cannot be used if the number of components is not 2')
        if (not(alpha_value) and alpha_selection=='manual'):
            raise ValueError('The the alpha_selection parameter is set to "manual", the alpha_value parameter must be provided')
        #you can't be plot or gui with non-2 components
        # Handle the plotting variables
        if (plot or gui):
            if active_labels is None:
                active_labels = np.ones(dataset.shape[0])
            self.active_labels = active_labels
            if colors is None:
                 self.colors = ['k','r','b','g','c']

        if gui:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.gridspec import GridSpec
            except ImportError:
                raise ImportError("Something wrong while loading matplotlib.pyplot! You probably don't have plotting libraries installed.")
            try:
                from ipywidgets import widgets, interact, Layout
                from IPython.display import display
            except ImportError:
                raise ImportError("To use the GUI, you must be running this code in a jupyter notebook that supports ipywidgets")

            transformed_data_auto, alphas_auto = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
            transformed_data_manual, alphas_manual = self.all_cpca(dataset, n_alphas, max_log_alpha)
            if (self.n_fg>1000):
                print("The GUI may be slow to respond with large numbers of data points. Consider using a subset of the original data.")

            """
            Handles the plotting
            """
            def graph_foreground(ax,fg, active_labels, alpha):
                for i, l in enumerate(np.sort(np.unique(active_labels))):
                    ax.scatter(fg[np.where(active_labels==l),0],fg[np.where(active_labels==l),1], color=self.colors[i%len(self.colors)], alpha=0.6)
                if (alpha==0):
                    ax.annotate(r'$\alpha$='+str(np.round(alpha,2))+" (PCA)", (0.05,0.05), xycoords='axes fraction')
                else:
                    ax.annotate(r'$\alpha$='+str(np.round(alpha,2)), (0.05,0.05), xycoords='axes fraction')


            """
            This code gets run whenever the widget slider is moved
            """
            def update(value):
                fig = plt.figure(figsize=[10,4])
                gs=GridSpec(2,4)

                for i in range(4):
                    ax1=fig.add_subplot(gs[int(i//2),i%2]) # First row, first column
                    fg = transformed_data_auto[i]
                    graph_foreground(ax1, fg, self.active_labels, alphas_auto[i])

                    ax5=fig.add_subplot(gs[:,2:]) # Second row, span all columns

                    alpha_idx = np.abs(alphas_manual-10**value).argmin()
                    fg = transformed_data_manual[alpha_idx]
                    graph_foreground(ax5, fg, self.active_labels, alphas_manual[alpha_idx])

                #if len(np.unique(self.active_labels))>1:
                    #plt.legend()

                plt.tight_layout()
                plt.show()

            widg = interact(update, value=widgets.FloatSlider(description=r'\(\log_{10}{\alpha} \)', min=-1, max=3, step=4/40, continuous_update=False, layout=Layout(width='80%')))

            return

        elif plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError("Something wrong while loading matplotlib.pyplot! You probably don't have plotting libraries installed.")
            if (alpha_selection=='auto'):
                transformed_data, best_alphas = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
                plt.figure(figsize=[14,3])
                for j, fg in enumerate(transformed_data):
                    plt.subplot(1,4,j+1)
                    for i, l in enumerate(np.sort(np.unique(self.active_labels))):
                        idx = np.where(self.active_labels==l)
                        plt.scatter(fg[idx,0],fg[idx,1], color=self.colors[i%len(self.colors)], alpha=0.6, label='Class '+str(i))
                    plt.title('Alpha='+str(np.round(best_alphas[j],2)))
                if len(np.unique(self.active_labels))>1:
                    plt.legend()
                plt.show()
            elif (alpha_selection=='manual'):
                transformed_data, best_alphas = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
                plt.figure(figsize=[14,3])
                for j, fg in enumerate(transformed_data):
                    plt.subplot(1,4,j+1)
                    for i, l in enumerate(np.sort(np.unique(self.active_labels))):
                        idx = np.where(self.active_labels==l)
                        plt.scatter(fg[idx,0],fg[idx,1], color=self.colors[i%len(self.colors)], alpha=0.6, label='Class '+str(i))
                    plt.title('Alpha='+str(np.round(best_alphas[j],2)))
                if len(np.unique(self.active_labels))>1:
                    plt.legend()
                plt.show()

            return

        else:
            if (alpha_selection=='auto'):
                transformed_data, best_alphas , v_top = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
                alpha_values = best_alphas
            elif (alpha_selection=='all'):
                transformed_data, all_alphas , v_top = self.all_cpca(dataset, n_alphas, max_log_alpha)
                alpha_values = all_alphas
            else:
                transformed_data = self.cpca_alpha(dataset, alpha_value)
                alpha_values = alpha_value
        if return_alphas:
            return transformed_data, alpha_values , v_top
        else:
            return transformed_data, v_top


    """
    This function performs contrastive PCA using the alpha technique on the
    active and background dataset. It automatically determines n_alphas=4 important values
    of alpha up to based to the power of 10^(max_log_alpha=5) on spectral clustering
    of the top subspaces identified by cPCA.
    The final return value is the data projected into the top (n_components = 2)
    subspaces, which can be plotted outside of this function
    """
    def automated_cpca(self, dataset, n_alphas_to_return, n_alphas, max_log_alpha):
        best_alphas, all_alphas, _, _ = self.find_spectral_alphas(n_alphas, max_log_alpha, n_alphas_to_return)
        best_alphas = np.concatenate(([0], best_alphas)) #one of the alphas is always alpha=0
        data_to_plot = []
        v_top_array = []
        for alpha in best_alphas:
            transformed_dataset,v_top = self.cpca_alpha(dataset=dataset, alpha=alpha)
            print("lelo")
            print(v_top.shape)
            v_top_array.append(v_top)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, best_alphas,v_top_array

    """
    This function performs contrastive PCA using the alpha technique on the
    active and background dataset. It returns the cPCA-reduced data for all values of alpha specified,
    both the active and background, as well as the list of alphas
    """
    def all_cpca(self, dataset, n_alphas, max_log_alpha):
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        data_to_plot = []
        v_top_array = []
        for alpha in alphas:
            transformed_dataset, v_top = self.cpca_alpha(dataset=dataset, alpha=alpha)
            v_top_array.append(v_top)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, alphas,v_top_array

    """
    Returns active and bg dataset projected in the cpca direction, as well as the top c_cpca eigenvalues indices.
    If specified, it returns the top_cpca directions
    """
    def cpca_alpha(self, dataset, alpha=1):
        n_components = self.n_components
        sigma = self.fg_cov - alpha*self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -n_components)[-n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:,eig_idx]
        reduced_dataset = dataset.dot(v_top)
        reduced_dataset[:,0] = reduced_dataset[:,0]*np.sign(reduced_dataset[0,0])
        reduced_dataset[:,1] = reduced_dataset[:,1]*np.sign(reduced_dataset[0,1])
        return reduced_dataset , v_top

    """
    This method performs spectral clustering on the affinity matrix of subspaces
    returned by contrastive pca, and returns (`=3) exemplar values of alpha
    """
    def find_spectral_alphas(self, n_alphas, max_log_alpha, n_alphas_to_return):
        self.create_affinity_matrix(max_log_alpha, n_alphas)
        affinity = self.affinity_matrix
        spectral = cluster.SpectralClustering(n_clusters=n_alphas_to_return, affinity='precomputed')
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        spectral.fit(affinity)
        labels = spectral.labels_
        best_alphas = list()
        for i in range(n_alphas_to_return):
            idx = np.where(labels==i)[0]
            if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
                affinity_submatrix = affinity[idx][:, idx]
                sum_affinities = np.sum(affinity_submatrix, axis=0)
                exemplar_idx = idx[np.argmax(sum_affinities)]
                best_alphas.append(alphas[exemplar_idx])
        return np.sort(best_alphas), alphas, affinity[0,:], labels

    """
    This method creates the affinity matrix of subspaces returned by contrastive pca
    """
    def create_affinity_matrix(self, max_log_alpha, n_alphas):
        from math import pi
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        subspaces = list()
        k = len(alphas)
        affinity = 0.5*np.identity(k) #it gets doubled
        for alpha in alphas:
            space , v_top = self.cpca_alpha(dataset=self.fg, alpha=alpha)
            q, r = np.linalg.qr(space)
            subspaces.append(q)
        for i in range(k):
            for j in range(i+1,k):
                q0 = subspaces[i]
                q1 = subspaces[j]
                u, s, v = np.linalg.svd(q0.T.dot(q1))
                affinity[i,j] = s[0]*s[1]
        affinity = affinity + affinity.T
        self.affinity_matrix = np.nan_to_num(affinity)



    def cpca_multi(self, lst , n_alphas = 40, max_log_alpha = 3, n_alphas_to_return =4):
        i = 0

        v_top_list = []
        proj_data_list = []
        for x in lst:

            lst.remove(x)
            self.fg = x
            self.bg = np.concatenate(lst,axis=0)
            lst.insert(i,x)
            projected_data, alpha1 , v_top = self.fit_transform(self.fg,self.bg,return_alphas=True)
            v_top_list.append(v_top[1])
            proj_data_list.append(projected_data[1])
            projected_data, alpha2 , v_top = self.fit_transform(self.bg,self.fg,return_alphas=True)
            v_top_list.append(v_top[1])
            proj_data_list.append(projected_data[1])

        v_top_list1 = np.array(v_top_list)
        proj_data_list1 = np.array(proj_data_list)

        return v_top_list1 , proj_data_list1

    def cpca_multi2(self,lst,fwd_ind,back_ind,n_alphas=40,max_log_alpha=3,n_alphas_to_return=4):
        
        self.fg = np.array(lst[fwd_ind[0]])
        for i in fwd_ind[1:]:
            self.fg = np.concatenate((self.fg,lst[i]),axis=0)

        self.bg = np.array(lst[back_ind[0]])
        for i in back_ind[1:]:
            self.bg = np.concatenate((self.bg,lst[i]),axis=0)

        projected_data,alpha1,v_top = self.fit_transform(self.fg,self.bg,return_alphas=True)
        return v_top,projected_data

  



    

