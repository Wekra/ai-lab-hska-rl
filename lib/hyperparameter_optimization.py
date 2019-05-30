import logging
import multiprocessing
import time
from functools import partial
from typing import List, Dict, Callable

import numpy as np
from sklearn.model_selection import ParameterGrid

LOGGER = logging.getLogger(__name__)


class GridSearch:

    def __init__(self,
                 grid_params: Dict[str, List],
                 fixed_params: Dict[str, List],
                 construct_env, construct_agent,
                 evaluation_func: Callable,
                 score_parameter: str,
                 grid_params_for_evaluation_func: List[str] = None):
        self.fixed_params = fixed_params
        self.grid = list(ParameterGrid(grid_params))
        self.final_scores = np.zeros(len(self.grid))
        self.construct_env = construct_env
        self.construct_agent = construct_agent
        self.evaluation_func = evaluation_func
        self.score_parameter = score_parameter
        self.grid_params_for_evaluation_func = grid_params_for_evaluation_func if grid_params_for_evaluation_func else []

    @staticmethod
    def _evaluate_single(args, n_runs, fixed_params, construct_env, construct_agent, evaluation_func,
                         score_parameter, grid_params_for_evaluation_func):
        index, params = args
        LOGGER.debug('Evaluating params: %s', params)
        params = {**params, **fixed_params}

        scores = []
        for i in range(n_runs):
            env = construct_env()
            agent = construct_agent(**params)
            grid_params = {x: params[x] for x in grid_params_for_evaluation_func}
            statistics = evaluation_func(env=env, agent=agent, **grid_params)
            score = [statistic[score_parameter] for statistic in statistics]
            scores.append(score)

        score = np.mean(scores)
        LOGGER.info('Finished evaluating set %s with score of %s.', index, score)
        return score

    def run(self, n_runs: int = 1):
        start_time = time.time()
        LOGGER.info('About to evaluate %s parameter sets.', len(self.grid))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        _ev_single = partial(self._evaluate_single, n_runs=n_runs, fixed_params=self.fixed_params,
                             construct_env=self.construct_env, construct_agent=self.construct_agent,
                             evaluation_func=self.evaluation_func,
                             score_parameter=self.score_parameter,
                             grid_params_for_evaluation_func=self.grid_params_for_evaluation_func)
        self.final_scores = pool.map(_ev_single, list(enumerate(self.grid)))

        LOGGER.info('Best parameter set was %s with score of %s',
                    self.grid[np.argmax(self.final_scores)], np.max(self.final_scores))
        LOGGER.info('Worst parameter set was %s  with score of %s',
                    self.grid[np.argmin(self.final_scores)], np.min(self.final_scores))
        LOGGER.info('Execution time: %s sec', time.time() - start_time)
