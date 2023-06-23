"""Modules for user response models, including human users."""
from typing import Dict, List, Union
import numpy as np
import scipy.special as ssp
from copy import deepcopy

from aprel.basics import Trajectory, TrajectorySet
from aprel.learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery, NLCommandQuery
from aprel.learning import QueryWithResponse, Demonstration, Preference, WeakComparison, FullRanking, NLCommand
import aprel


class User:
    """
    An abstract class to model the user of which the reward function is being learned.
    
    Parameters:
        params_dict (Dict): parameters of the user model.
    """
    def __init__(self, params_dict: Dict = None):
        if params_dict is not None:
            self._params = params_dict.copy()
        else:
            self._params = {}
    
    @property
    def params(self):
        """Returns the parameters of the user."""
        return self._params
        
    @params.setter
    def params(self, params_dict: Dict):
        """Replaces the parameters of the user if new values are provided."""
        params_dict_copy = params_dict.copy()
        for key, value in self._params.items():
            params_dict_copy.setdefault(key, value)
        self._params = params_dict_copy
        
    def copy(self):
        return deepcopy(self)
        
    def response_logprobabilities(self, query: Query) -> np.array:
        """
        Returns the log probability for each response in the response set for the query under the user.
        
        Args:
            query (Query): The query for which the log-probabilites are being calculated.
            
        Returns:
            numpy.array: An array, where each entry is the log-probability of the corresponding response
                in the :py:attr:`query`'s response set.
        """
        raise NotImplementedError
        
    def response_probabilities(self, query: Query) -> np.array:
        """
        Returns the probability for each response in the response set for the query under the user.
        
        Args:
            query (Query): The query for which the probabilites are being calculated.
            
        Returns:
            numpy.array: An array, where each entry is the probability of the corresponding response in
                the :py:attr:`query`'s response set.
        """
        return np.exp(self.response_logprobabilities(query))
        
    def loglikelihood(self, data: QueryWithResponse) -> float:
        """
        Returns the loglikelihood of the given user feedback under the user.
        This just uses the response_logprobabilities method.
        
        Args:
            data (QueryWithResponse): The data (which keeps a query and a response) for which the
                loglikelihood is going to be calculated.
            
        Returns:
            float: The loglikelihood of :py:attr:`data` under the user.
        """
        logprobs = self.response_logprobabilities(data)
        if isinstance(data, Preference) or isinstance(data, WeakComparison):
            idx = np.where(data.query.response_set == data.response)[0][0]
        # TODO: add case for NLCommand (?)
        elif isinstance(data, NLCommand):
            raise NotImplementedError
        elif isinstance(data, FullRanking):
            idx = np.where((data.query.response_set == data.response).all(axis=1))[0][0]
        return logprobs[idx]

    def likelihood(self, data: QueryWithResponse) -> float:
        """
        Returns the likelihood of the given user feedback under the user.
        
        Args:
            data (QueryWithResponse): The data (which keeps a query and a response) for which the
                likelihood is going to be calculated.
            
        Returns:
            float: The likelihood of :py:attr:`data` under the user.
        """
        return np.exp(self.loglikelihood(data))
        
    def loglikelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """
        Returns the loglikelihood of the given feedback dataset under the user.
        
        Args:
            dataset (List[QueryWithResponse]): The dataset (which keeps a list of feedbacks) for which the
                loglikelihood is going to be calculated.
            
        Returns:
            float: The loglikelihood of :py:attr:`dataset` under the user.
        """
        return np.sum([self.loglikelihood(data) for data in dataset])
        
    def likelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """
        Returns the likelihood of the given feedback dataset under the user.
        
        Args:
            dataset (List[QueryWithResponse]): The dataset (which keeps a list of feedbacks) for which the
                likelihood is going to be calculated.
            
        Returns:
            float: The likelihood of :py:attr:`dataset` under the user.
        """
        return np.exp(self.loglikelihood_dataset(dataset))

    # TODO: update this to be compatible with not necessarily responding from a response set?
    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """
        Simulates the user's responses to the given queries.
        
        Args:
            queries (Query or List[Query]): A query or a list of queries for which the user's response(s)
                is/are requested.
                
        Returns:
            List: A list of user responses where each response corresponds to the query in the :py:attr:`queries`.
                :Note: The return type is always a list, even if the input is a single query.
        """
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            probs = self.response_probabilities(query)
            idx = np.random.choice(len(probs), p=probs)
            responses.append(query.response_set[idx])
        return responses


class SoftmaxUser(User):
    """
    Softmax user class whose response model follows the softmax choice rule, i.e., when presented with multiple
    trajectories, this user chooses each trajectory with a probability that is proportional to the expontential of
    the reward of that trajectory.
    
    Parameters:
        params_dict (Dict): the parameters of the softmax user model, which are:
            - `weights` (numpy.array): the weights of the linear reward function.
            - `beta` (float): rationality coefficient for comparisons and rankings.
            - `beta_D` (float): rationality coefficient for demonstrations.
            - `delta` (float): the perceivable difference parameter for weak comparison queries.

    Raises:
        AssertionError: if a `weights` parameter is not provided in the :py:attr:`params_dict`.
    """
    def __init__(self, params_dict: Dict):
        assert('weights' in params_dict), 'weights is a required parameter for the softmax user model.'       
        params_dict_copy = params_dict.copy()
        params_dict_copy.setdefault('beta', 1.0)
        params_dict_copy.setdefault('beta_D', 1.0)
        params_dict_copy.setdefault('delta', 0.1)
        
        super(SoftmaxUser, self).__init__(params_dict_copy)
        
    def response_logprobabilities(self, query: Query) -> np.array:
        """Overwrites the parent's method. See :class:`.User` for more information."""
        if isinstance(query, PreferenceQuery):
            rewards = self.params['beta'] * self.reward(query.slate)
            return rewards - ssp.logsumexp(rewards)

        elif isinstance(query, NLCommandQuery):
            d = len(self.params['weights'])

            # As an ad hoc solution, we'll just sample a bunch of random xf's in the unit ball (norm=1)
            # in order to model the different responses we may get (since we don't have a fixed response set).
            num_xf_samples = 10
            xfs = [aprel.util_funs.get_random_normalized_vector(d) for _ in range(num_xf_samples)]

            logprobs = np.zeros(num_xf_samples)

            # Find ideal trajectory's features \phi star for log likelihood computation
            ideal_trajectory = None
            ideal_reward = -np.inf
            assert 'trajectory_set' in self.params
            for trajectory in self.params['trajectory_set']:
                r = self.reward(trajectory)
                if r > ideal_reward:
                    ideal_reward = r
                    ideal_trajectory = trajectory

            # Calculation of matrix A
            feature_diff = ideal_trajectory.features - query.slate[0].features
            A = np.expand_dims(self.params['weights'], axis=-1) @ np.expand_dims(feature_diff, axis=-1).T

            # Monte Carlo estimate of surface integral (denominator)
            num_monte_carlo_samples = 100
            X = np.random.randn(num_monte_carlo_samples, d)
            X = X / np.linalg.norm(X, axis=-1)
            integrand = np.mean(np.exp(np.sum((X @ A) * X, axis=-1)))
            surface_area = 2 * np.pi**(d / 2) / ssp.gamma(d / 2)
            denominator = surface_area * integrand
            logdenominator = np.log(denominator)
            assert np.isscalar(logdenominator)

            for i, xf in enumerate(xfs):
                xf = np.expand_dims(xf, axis=-1)
                lognumerator = xf.T @ A @ xf
                assert np.isscalar(lognumerator)
                logprobs[i] = lognumerator - logdenominator
            return logprobs

        elif isinstance(query, WeakComparisonQuery):
            rewards = self.params['beta'] * self.reward(query.slate)
            logprobs = np.zeros((3))
            logprobs[1] = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            logprobs[2] = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            logprobs[0] = np.log(np.exp(2*self.params['delta']) - 1) + logprobs[1] + logprobs[2]
            return logprobs
            
        elif isinstance(query, FullRankingQuery):
            rewards = self.params['beta'] * self.reward(query.slate)
            logprobs = np.zeros(len(query.response_set))
            for response_id in range(len(query.response_set)):
                response = query.response_set[response_id]
                sorted_rewards = rewards[response]
                logprobs[response_id] = np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(response))])
            return logprobs
        raise NotImplementedError("response_logprobabilities is not defined for demonstration queries.")

    # Note: There isn't any reason why loglikelihood does not call response_logprobabilities.
    # In the parent User class, loglikelihood does just call response_logprobabilities.
    def loglikelihood(self, data: QueryWithResponse) -> float:
        """
        Overwrites the parent's method. See :class:`.User` for more information.
        
        :Note: The loglikelihood value is the logarithm of the `unnormalized` likelihood if the
            input is a demonstration. Otherwise, it is the exact loglikelihood.
        """
        if isinstance(data, Demonstration):
            return self.params['beta_D'] * self.reward(data)
        
        elif isinstance(data, Preference):
            rewards = self.params['beta'] * self.reward(data.query.slate)
            return rewards[data.response] - ssp.logsumexp(rewards)

        elif isinstance(data, NLCommand):
            d = len(self.params['weights'])
            xf = data.response / np.linalg.norm(data.response)

            # Find ideal trajectory's features \phi star for log likelihood computation
            ideal_trajectory = None
            ideal_reward = -np.inf
            assert 'trajectory_set' in self.params
            for trajectory in self.params['trajectory_set']:
                r = self.reward(trajectory)
                if r > ideal_reward:
                    ideal_reward = r
                    ideal_trajectory = trajectory

            feature_diff = ideal_trajectory.features - data.query.slate[0].features
            A = np.expand_dims(self.params['weights'], axis=-1) @ np.expand_dims(feature_diff, axis=-1).T

            xf = np.expand_dims(xf, axis=-1)
            lognumerator = xf.T @ A @ xf
            assert np.isscalar(lognumerator)

            # Monte Carlo estimate of surface integral (denominator)
            num_monte_carlo_samples = 100
            X = np.random.randn(num_monte_carlo_samples, d)
            X = X / np.linalg.norm(X, axis=-1)
            integrand = np.mean(np.exp(np.sum((X @ A) * X, axis=-1)))
            surface_area = 2 * np.pi**(d / 2) / ssp.gamma(d / 2)
            denominator = surface_area * integrand
            logdenominator = np.log(denominator)
            assert np.isscalar(logdenominator)

            return lognumerator - logdenominator
            
        elif isinstance(data, WeakComparison):
            rewards = self.params['beta'] * self.reward(data.query.slate)
            
            logp0 = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            if data.response == 0: return logp0
            
            logp1 = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            if data.response == 1: return logp1
            
            if data.response == -1:
                return np.log(np.exp(2*self.params['delta']) - 1) + logp0 + logp1
                
        elif isinstance(data, FullRanking):
            rewards = self.params['beta'] * self.reward(data.query.slate)
            sorted_rewards = rewards[data.response]
            return np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(data.response))])
            
        raise NotImplementedError("User response model for the given data is not implemented.")

    def reward(self, trajectories: Union[Trajectory, TrajectorySet]) -> Union[float, np.array]:
        """
        Returns the reward of a trajectory or a set of trajectories conditioned on the user.
        
        Args:
            trajectories (Trajectory or TrajectorySet): The trajectories for which the reward will be calculated.
            
        Returns:
            numpy.array or float: the reward value of the :py:attr:`trajectories` conditioned on the user.
        """
        if isinstance(trajectories, TrajectorySet):
            return np.dot(trajectories.features_matrix, self.params['weights'])
        return np.dot(trajectories.features, self.params['weights'])


class CustomFeatureUser(SoftmaxUser):
    """
    CustomFeatureUser is a Softmax User, but instead performs reward computations based on a feature function
    that is different from the one already specified by the environment (Trajectory.features).

    Parameters:
        params_dict (Dict): the parameters of the softmax user model, which are:
            - `feature_func` (function): List of state-action tuples -> np.array of feature values

    Raises:
        AssertionError: if a `feature_func` parameter is not provided in the :py:attr:`params_dict`.
    """

    def __init__(self, params_dict: Dict):
        assert ('feature_func' in params_dict), 'feature_func is a required parameter for the custom feature user model.'
        params_dict_copy = params_dict.copy()

        super(CustomFeatureUser, self).__init__(params_dict_copy)

    def reward(self, trajectories: Union[Trajectory, TrajectorySet]) -> Union[float, np.array]:
        # TODO: Can try to pre-compute these values and store them in the
        #  Trajectory/TrajectorySet for speed up if necessary
        if isinstance(trajectories, TrajectorySet):
            features_matrix = np.array([self.params['feature_func'](trajectory.trajectory) for trajectory in trajectories])
            return np.dot(features_matrix, self.params['weights'])
        else:
            features = self.params['feature_func'](trajectories.trajectory)
            return np.dot(features, self.params['weights'])


class HumanUser(User):
    """
    Human user class whose response model is unknown. This class is useful for interactive runs, where
    a real human responds to the queries rather than simulated user models.
    
    Parameters:
        delay (float): The waiting time between each trajectory visualization during querying in seconds.
        
    Attributes:
        delay (float): The waiting time between each trajectory visualization during querying in seconds.
    """
    def __init__(self, delay: float = 0.):
        super(HumanUser, self).__init__()
        self.delay = delay
        
    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """
        Interactively asks for the user's responses to the given queries.
        
        Args:
            queries (Query or List[Query]): A query or a list of queries for which the user's response(s)
                is/are requested.
                
        Returns:
            List: A list of user responses where each response corresponds to the query in the :py:attr:`queries`.
                :Note: The return type is always a list, even if the input is a single query.
        """
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            responses.append(query.visualize(self.delay))
        return responses
