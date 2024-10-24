import numpy as np
import neurogym as ngym
from neurogym import spaces
import pandas as pd
import copy

import general.utility as u


class Task(ngym.TrialEnv):    
    def sample_trials(self, n_trls, **kwargs):
        info = []
        inputs = []
        targs = []
        for i in range(n_trls):
            info_i = copy.deepcopy(self.new_trial(**kwargs))
            inp = copy.deepcopy(self.ob)
            targ = copy.deepcopy(self.gt)
            inputs.append(inp)
            targs.append(targ)
            info.append(info_i)
        info = pd.DataFrame(info)
        return info, inputs, targs


class CuedContinuousReportTask(Task):
    """Retrospectively cued task with report of a continuous stimulus."""

    metadata = {
        "paper_link": "https://www.nature.com/articles/s41586-021-03390-w",
        "paper_name": """Shared mechanisms underlie the control of working
        memory and attention""",
        "tags": ["perceptual", "context dependent", "two-alternative", "supervised"],
    }

    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        sigma=0,
        n_cols=64,
        periods=None,
    ):
        super().__init__(dt=dt)

        # trial conditions
        self.periods = periods
        self.cues = [0, 1]  # index for context inputs
        # color responses
        self.possible_colors = np.linspace(0, np.pi * 2, n_cols + 1)[:-1]
        self.choices = u.radian_to_sincos(self.possible_colors)
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.fix_thr = 0.9
        self.resp_thr = 0.9
        self.reward_thr = 0.3

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,
            "stimulus": 500,
            "delay1": ngym.random.TruncExp(600, 500, 1000),
            "cue": 300,
            "delay2": ngym.random.TruncExp(600, 500, 700),
            "decision": 500,
        }
        if timing is not None:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = [
            "fixation",
            "color1_sin",
            "color1_cos",
            "color2_sin",
            "color2_cos",
            "cue_1",
            "cue_2",
        ]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(names),), dtype=np.float32, name=name
        )
        self.obs_dims = len(names)

        name = {
            "eye_x": 0,
            "eye_y": 1,
        }
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(name),), name=name)
        self.action_dims = len(name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            "color1": self.rng.choice(self.possible_colors),
            "color2": self.rng.choice(self.possible_colors),
            "cue": self.rng.choice(self.cues),
        }
        trial.update(kwargs)

        color1, color2 = trial["color1"], trial["color2"]
        c1_sin, c1_cos = u.radian_to_sincos(color1)
        c2_sin, c2_cos = u.radian_to_sincos(color2)

        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        self.add_period(self.periods)

        self.add_ob(1, where="fixation")
        self.set_ob(0, period="decision", where="fixation")
        self.add_ob(c1_sin, period="stimulus", where="color1_sin")
        self.add_ob(c1_cos, period="stimulus", where="color1_cos")
        self.add_ob(c2_sin, period="stimulus", where="color2_sin")
        self.add_ob(c2_cos, period="stimulus", where="color2_cos")
        self.add_randn(0, self.sigma)

        if trial["cue"] == 1:
            self.add_ob(1, period="cue", where="cue_1")
            tc_sin, tc_cos = u.radian_to_sincos(color1)
            dc_sin, dc_cos = u.radian_to_sincos(color2)
            trial["target_color"] = color1
            trial["distractor_color"] = color2
        else:
            self.add_ob(1, period="cue", where="cue_2")
            tc_sin, tc_cos = u.radian_to_sincos(color2)
            dc_sin, dc_cos = u.radian_to_sincos(color1)
            trial["target_color"] = color2
            trial["distractor_color"] = color1

        self.set_groundtruth((tc_sin, tc_cos), period="decision")
        trial["start_ind"] = self.start_ind
        trial["end_ind"] = self.end_ind

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        action_dev = np.sqrt(np.sum(action**2))
        new_trial = False
        reward = 0
        if self.in_period("fixation"):
            if action_dev > self.fix_thr:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            if action_dev > self.resp_thr:  # broke fixation
                new_trial = True
                if np.sqrt(np.sum((action - gt) ** 2)) < self.reward_thr:
                    reward = self.rewards["correct"]
                    self.performance = 1

        return ob, reward, False, {"new_trial": new_trial, "gt": gt}


class RetrospectiveContinuousReportTask(CuedContinuousReportTask):
    def __init__(self, *args, **kwargs):
        periods = ["fixation", "stimulus", "delay1", "cue", "delay2", "decision"]
        super().__init__(*args, periods=periods, **kwargs)


class ProspectiveContinuousReportTask(CuedContinuousReportTask):
    def __init__(self, *args, **kwargs):
        periods = ["fixation", "cue", "delay1", "stimulus", "delay2", "decision"]
        super().__init__(*args, periods=periods, **kwargs)


class DelayedMatchTask(Task):
    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        sigma=0,
        n_dirs=16,
        match_func=None,
        body=False,
    ):
        super().__init__(dt=dt)
        self.decide_match = match_func
        self.body = body
        self.periods = ["fixation", "sample", "delay", "test"]
        self.possible_directions = np.linspace(0, np.pi * 2, n_dirs + 1)[:-1]
        self.sigma = sigma / np.sqrt(self.dt)
        self.fix_thr = 0.9
        self.resp_thr = 0.9
        self.reward_thr = 0.3
        self.rewards = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,
            "sample": 500,
            "delay": ngym.random.TruncExp(600, 500, 1000),
            "test": 500,
        }
        if timing is not None:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = [
            "fixation",
            "dir_sin",
            "dir_cos",
        ]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(names),), dtype=np.float32, name=name
        )
        self.obs_dims = len(names)

        name = {
            "lever": 0,
        }
        if self.body:
            name["eye_x"] = 1
            name["eye_y"] = 2
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(name),), name=name)
        self.action_dims = len(name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            "sample": self.rng.choice(self.possible_directions),
            "test": self.rng.choice(self.possible_directions),
        }
        trial.update(kwargs)

        sample, test = trial["sample"], trial["test"]
        sample_sin, sample_cos = u.radian_to_sincos(sample)
        test_sin, test_cos = u.radian_to_sincos(test)
        match = self.decide_match(sample, test)

        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        self.add_period(self.periods)

        self.add_ob(1, where="fixation")
        self.set_ob(0, period="test", where="fixation")
        self.add_ob(sample_sin, period="sample", where="dir_sin")
        self.add_ob(sample_cos, period="sample", where="dir_cos")
        self.add_ob(test_sin, period="test", where="dir_sin")
        self.add_ob(test_cos, period="test", where="dir_cos")
        self.add_randn(0, self.sigma)

        target = (1 - match,)
        if self.body:
            target = target + (0, 0)
        self.set_groundtruth(target, period="test")
        trial["start_ind"] = self.start_ind
        trial["end_ind"] = self.end_ind

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        action_dev = np.sqrt(np.sum(action**2))
        new_trial = False
        reward = 0
        if self.in_period("fixation"):
            if action_dev > self.fix_thr:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            if action_dev > self.resp_thr:  # broke fixation
                new_trial = True
                if np.sqrt(np.sum((action - gt) ** 2)) < self.reward_thr:
                    reward = self.rewards["correct"]
                    self.performance = 1

        return ob, reward, False, {"new_trial": new_trial, "gt": gt}


class DelayedMatchToSample(DelayedMatchTask):
    def __init__(self, *args, **kwargs):
        def match_func(sample, test):
            return sample == test
        super().__init__(*args, match_func=match_func, **kwargs)


def categorize(sample, cat_dir):
    return np.sign(u.normalize_periodic_range(sample - cat_dir))

        
class DelayedMatchToCategory(DelayedMatchTask):
    def __init__(self, *args, cat_dir=None, n_dirs=16, **kwargs):
        if cat_dir is None:
            cat_dir = 1 / (2 * n_dirs)
        def match_func(sample, test):
            s_sign = categorize(sample, cat_dir)
            t_sign = categorize(test, cat_dir)
            return s_sign == t_sign
        super().__init__(*args, match_func=match_func, n_dirs=n_dirs, **kwargs)
        self.cat_dir = cat_dir

