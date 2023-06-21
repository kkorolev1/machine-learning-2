import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from distributions import BaseDistribution, NumericalDistribution, CategoricalDistribution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import KernelDensity, NearestNeighbors

class BaseOptimizer(BaseEstimator, ABC):
    '''
    A base class for all hyperparameter optimizers
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, n_jobs: Optional[int] = None,
                 verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - estimator: sklearn model instance
          - param_distributions: a dictionary of parameter distributions,
            e.g. param_distributions['num_epochs'] = IntUniformDistribution(100, 200)
          - scoring: sklearn scoring object, see
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            if left None estimator must have 'score' attribute
          - cv: number of folds to cross-validate
          - num_runs: number of iterations to fit hyperparameters
          - num_dry_runs: number of dry runs (i.e. random strategy steps) to gather initial statistics
          - num_samples_per_run: number of hyperparameters set to sample each iteration
          - n_jobs: number of parallel processes to fit algorithms
          - verbose: whether to print debugging information (you can configure debug as you wish)
          - random_state: RNG seed to control reproducibility
        '''
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.num_runs = num_runs
        self.num_samples_per_run = num_samples_per_run
        self.num_dry_runs = num_dry_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.reset()

    def reset(self):
        '''
        Reset fields used for fitting
        '''
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.array([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])
        np.random.seed(self.random_state)

    def sample_params(self) -> Dict[str, np.array]:
        '''
        Sample self.num_samples_per_run set of hyperparameters
        Returns:
          - sampled_params: dict of arrays of parameter samples,
            e.g. sampled_params['num_epochs'] = np.array([178, 112, 155])
        '''
        sampled_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        for param_name, dist in self.param_distributions.items():
            sampled_params[param_name] = dist.sample(self.num_samples_per_run)
        return sampled_params

    @abstractmethod
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        msg = f'method \"select_params\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def cross_validate(self, X: np.array, y: Optional[np.array],
                       params: Dict[str, Any]) -> float:
        '''
        Calculate cross-validation score for a set of params
        Consider using estimator.set_params() and sklearn.model_selection.cross_validate()
        Also use self.splitter as a cv parameter in cross_validate
        Params:
          - X: object features
          - y: object labels
          - params: a set of params to score
        Returns:
          - score: mean cross-validation score
        '''
        # Your code here (⊃｡•́‿•̀｡)⊃
        estimator = self.estimator.set_params(**params)
        if hasattr(self.estimator, 'random_state'):
            self.estimator.random_state = self.random_state
        cross_val_scores = cross_validate(estimator, X, y, scoring=self.scoring, cv=self.splitter, n_jobs=self.n_jobs, verbose=False)
        return cross_val_scores["test_score"].mean()

    def _update_history(self, new_params, new_score):
        self.scores_history = np.append(self.scores_history, new_score)

        for param_name, param_values in self.params_history.items():
            self.params_history[param_name] = np.append(param_values, new_params[param_name])

    def _update_best(self, new_params, new_score):
        if self.best_score is None or new_score > self.best_score:
            self.best_score = new_score
            self.best_params = new_params
            self.best_estimator = self.estimator.set_params(**new_params)

    def fit(self, X_train: np.array, y_train: Optional[np.array] = None) -> BaseEstimator:
        '''
        Find the best set of hyperparameters with a specific search strategy
        using cross-validation and fit self.best_estimator on whole training set
        Params:
          - X_train: array of train features of shape (num_samples, num_features)
          - y_train: array of train labels of shape (num_samples, )
            if left None task is unsupervised
        Returns:
          - self: (sklearn standard convention)
        '''
        self.reset()
        if y_train is not None and np.issubdtype(y_train.dtype, np.integer):
            self.splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)

        # Your code here (⊃｡•́‿•̀｡)⊃
        for run_index in range(self.num_runs):
            new_params = self.select_params(self.params_history, self.scores_history, self.sample_params())
            new_score = self.cross_validate(X_train, y_train, new_params)
            self._update_history(new_params, new_score)
            self._update_best(new_params, new_score)

            if self.verbose:
                print(f"RUN {run_index + 1}/{self.num_runs}. Score: {new_score}. Params: {new_params}. Best score: {self.best_score}. Params: {self.best_params}\n")

        self.best_estimator = self.estimator.set_params(**self.best_params)
        if hasattr(self.best_estimator, 'random_state'):
            self.best_estimator.random_state = self.random_state
        self.best_estimator = self.best_estimator.fit(X_train, y_train)

        return self

    def predict(self, X_test: np.array) -> np.array:
        '''
        Generate a prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test predictions of shape (num_samples, )
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        # your code here ┐(シ)┌
        return self.best_estimator.predict(X_test)

    def predict_proba(self, X_test: np.array) -> np.array:
        '''
        Generate a probability prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test probabilities of shape (num_samples, num_classes)
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        if not hasattr(self.best_estimator, 'predict_proba'):
            raise ValueError('Estimator does not support predict_proba')

        # your code here ┐(シ)┌
        return self.best_estimator.predict_proba(X_test)

class RandomSearchOptimizer(BaseOptimizer):
    '''
    An optimizer implementing random search strategy
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        super().__init__(
            estimator, param_distributions, scoring, cv=cv,
            num_runs=num_runs, num_dry_runs=0, num_samples_per_run=1,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        new_params = {param: np.random.choice(values) for param, values in sampled_params.items()}
        return new_params

class GPOptimizer(BaseOptimizer):
    '''
    An optimizer implementing gaussian process strategy
    '''
    @staticmethod
    def calculate_expected_improvement(y_star: float, mu: np.array,
                                       sigma: np.array) -> np.array:
        '''
        Calculate EI values for passed parameters of normal distribution
        hint: consider using scipy.stats.norm
        Params:
          - y_star: optimal (maximal) score value
          - mu: array of mean values of normal distribution of size (num_samples_per_run, )
          - sigma: array of std values of normal distribution of size (num_samples_per_run, )
        Retuns:
          - ei: array of EI values of size (num_samples_per_run, )
        '''
        # Your code here (⊃｡•́‿•̀｡)⊃
        z = (mu - y_star) / sigma
        return (mu - y_star) * norm.cdf(z) + sigma * norm.pdf(z)

    def select_categories(self, params_history: Dict[str, np.array], scores_history: np.array, y_star: float):
        new_params = {}

        for param_name in params_history:
            dist = self.param_distributions[param_name]
            if isinstance(dist, NumericalDistribution):
                continue
            
            mus = []
            sigmas = []

            param_history = params_history[param_name]

            for category in dist.categories:
                scores_for_category = scores_history[param_history == category]

                if len(scores_for_category) == 0:
                    mus.append(y_star)
                    sigmas.append(1)
                else:
                    mus.append(scores_for_category.mean())
                    sigma2 = (1 + ((scores_for_category - mus[-1])**2).sum()) / (1 + len(scores_for_category))
                    sigmas.append(np.sqrt(sigma2))

            ei = GPOptimizer.calculate_expected_improvement(y_star, np.array(mus), np.array(sigmas))
            new_params[param_name] = dist.categories[ei.argmax()]
        
        return new_params

    def scale_params(self, params: Dict[str, np.array]):
        scaled_params = []

        for param_name in params:
            dist = self.param_distributions[param_name]

            if isinstance(dist, NumericalDistribution):
                scaled_params.append(dist.scale(params[param_name]))
        
        return np.array(scaled_params).T

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        # Your code here (⊃｡•́‿•̀｡)⊃
        if len(scores_history) < self.num_dry_runs:
            new_params = {}
            for param_name in sampled_params:
                new_params[param_name] = np.random.choice(sampled_params[param_name])
            
            return new_params
        
        y_star = np.max(scores_history)
        new_params = self.select_categories(params_history, scores_history, y_star)
        
        kernel = ConstantKernel() + WhiteKernel() + RBF()

        num_numerical = 0
        for param_name in params_history:
            if hasattr(self.param_distributions[param_name], "scale"):
                num_numerical += 1
        
        if num_numerical > 0:
            scaled_params_history = self.scale_params(params_history)      
            gp = GaussianProcessRegressor(kernel).fit(scaled_params_history, scores_history)
            
            scaled_sampled_params = self.scale_params(sampled_params)
            mean, std = gp.predict(scaled_sampled_params, return_std=True)
            
            ei = GPOptimizer.calculate_expected_improvement(y_star, mean, std)
            optimal_index = np.argmax(ei)

            for param_name, param_values in sampled_params.items():
                if isinstance(self.param_distributions[param_name], NumericalDistribution):
                    new_params[param_name] = param_values[optimal_index]

        return new_params

class TPEOptimizer(BaseOptimizer):
    '''
    An optimizer implementing tree-structured Parzen estimator strategy
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, gamma: float = 0.75,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - gamma: scores quantile used for history splitting
        '''
        super().__init__(
            estimator, param_distributions, scoring, cv=cv, num_runs=num_runs,
            num_dry_runs=num_dry_runs, num_samples_per_run=num_samples_per_run,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
        self.gamma = gamma

    @staticmethod
    def estimate_log_density(scaled_params_history: np.array,
                             scaled_sampled_params: np.array, bandwidth: float):
        '''
        Estimate log density of sampled numerical hyperparameters based on
        numerical hyperparameters history subset
        Params:
          - scaled_params_history: array of scaled numerical hyperparameters history subset
            of size (subset_size, num_numerical_params)
          - scaled_sampled_params: array of scaled sampled numerical hyperparameters
            of size (num_samples_per_run, num_numerical_params)
          - bandwidth: bandwidth for KDE
        Returns:
          - log_density: array of estimated log probabilities of size (num_samples_per_run, )
        '''
        # Your code here (⊃｡•́‿•̀｡)⊃
        return KernelDensity(bandwidth=bandwidth).fit(scaled_params_history).score_samples(scaled_sampled_params)

    def scale_params(self, params: Dict[str, np.array]):
        scaled_params = []

        for param_name in params:
            dist = self.param_distributions[param_name]

            if isinstance(dist, NumericalDistribution):
                scaled_params.append(dist.scale(params[param_name]))
        
        return np.array(scaled_params).T
        
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        # Your code here (⊃｡•́‿•̀｡)⊃
        if len(scores_history) < self.num_dry_runs:
            new_params = {}
            for param_name in sampled_params:
                new_params[param_name] = np.random.choice(sampled_params[param_name])
            
            return new_params
        
        y_star = np.max(scores_history)

        num_numerical = 0
        for param_name in params_history:
            if hasattr(self.param_distributions[param_name], "scale"):
                num_numerical += 1
        
        if num_numerical > 0:
            scaled_params_history = self.scale_params(params_history)
            scaled_sampled_params = self.scale_params(sampled_params)

            q = np.quantile(scores_history, self.gamma)
            bandwidth = np.median(NearestNeighbors(n_neighbors=1).fit(scaled_params_history).kneighbors(scaled_sampled_params))

            log_l = self.estimate_log_density(scaled_params_history[scores_history < q], scaled_sampled_params, bandwidth)
            log_g = self.estimate_log_density(scaled_params_history[scores_history >= q], scaled_sampled_params, bandwidth)

            optimal_index = np.argmax(log_g - log_l)

            for param_name, param_values in sampled_params.items():
                if isinstance(self.param_distributions[param_name], NumericalDistribution):
                    new_params[param_name] = param_values[optimal_index]

