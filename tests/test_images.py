"""
Image observations: the Atari preprocessing wrapper and the CNN path of every
learner, on a fake two-player "Atari" game so the tests need no ROMs.

The real Space Invaders preset is exercised only when its ROMs are installed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parse_env  # noqa: E402
from sepsro.envs import make_env  # noqa: E402
from sepsro.envs.base import EnvSpec, MultiAgentEnv, TimeStep  # noqa: E402
from sepsro.policies import RandomPolicy  # noqa: E402
from sepsro.rollout import play_episode  # noqa: E402

torch = pytest.importorskip('torch')

from sepsro.envs.wrappers import AtariPreprocessing  # noqa: E402


class FakeAtari(MultiAgentEnv):
    """Two players, 210x160 grey frames whose pixels equal the frame counter;
    reward 1 to each player per frame; ends after `length` frames."""

    def __init__(self, length=40):
        super().__init__()
        self.length = length
        agents = ('first_0', 'second_0')
        self.spec = EnvSpec(name='fake_atari', backend='pettingzoo', agents=agents,
                            num_actions={a: 6 for a in agents},
                            obs_shape={a: (210, 160, 1) for a in agents}, turn_based=False)

    def _frame(self):
        return {a: np.full((210, 160, 1), self.frame % 256, np.float32) for a in self.spec.agents}

    def _reset(self, seed):
        self.frame = 0
        return TimeStep(self._frame(), {a: np.ones(6, bool) for a in self.spec.agents},
                        self.spec.agents, {a: 0.0 for a in self.spec.agents})

    def _step(self, actions):
        assert set(actions) == set(self.spec.agents)
        self.frame += 1
        done = self.frame >= self.length
        return TimeStep(self._frame(), {} if done else {a: np.ones(6, bool) for a in self.spec.agents},
                        () if done else self.spec.agents, {a: 1.0 for a in self.spec.agents},
                        terminated=done)


def test_atari_wrapper_skip_pool_resize_stack():
    env = AtariPreprocessing(FakeAtari(length=10), frame_skip=4, stack=4, size=84)
    assert env.spec.obs_shape['first_0'] == (4, 84, 84)
    ts = env.reset(seed=0)
    obs = ts.observations['first_0']
    assert obs.dtype == np.uint8 and obs.shape == (4, 84, 84)
    assert (obs == 0).all()                         # stack filled with the first frame
    ts = env.step({a: 0 for a in env.spec.agents})
    assert ts.rewards == {'first_0': 4.0, 'second_0': 4.0}   # 4 frames' rewards summed
    newest = ts.observations['first_0'][-1]
    assert (newest == 4).all()                      # max of frames 3 and 4
    assert (ts.observations['first_0'][:-1] == 0).all()
    ts = env.step({a: 0 for a in env.spec.agents})
    ts = env.step({a: 0 for a in env.spec.agents})  # frames 9..10: ends after 2 of 4
    assert ts.terminated and ts.rewards['first_0'] == 2.0


def test_atari_wrapper_truncation_counts_decisions():
    env = AtariPreprocessing(FakeAtari(length=1000), max_episode_steps=3)
    ep = play_episode(env, {a: RandomPolicy() for a in env.spec.agents}, np.random.default_rng(0))
    assert ep.length == 3 and ep.truncated


def _image_episode(learner_policy, agents):
    env = AtariPreprocessing(FakeAtari(length=24))
    joint = {agents[0]: learner_policy, agents[1]: RandomPolicy()}
    return env, play_episode(env, joint, np.random.default_rng(0), record=True)


@pytest.mark.parametrize('kind', ['pg', 'dqn'])
def test_rl_learners_train_on_images(kind):
    from sepsro.oracles import make_learner_factory
    from config import get_config, resolve
    args = resolve(get_config().parse_args(['--env', 'rps', '--pg-hidden', '16',
                                            '--dqn-hidden', '16', '--dqn-batch-size', '4',
                                            '--dqn-min-buffer-size-to-learn', '4',
                                            '--dqn-learn-every', '1',
                                            '--dqn-replay-buffer-capacity', '64']))
    spec = AtariPreprocessing(FakeAtari()).spec
    learner = make_learner_factory(kind, spec, args)('first_0', seed=0)
    env, ep = _image_episode(learner.acting_policy(), spec.agents)
    before = [p.detach().clone() for p in learner.net.parameters()]
    for _ in range(8):
        learner.end_episode(ep.trajectories['first_0'])
    learner.end_episode(ep.trajectories['first_0'])
    changed = any((a != b).any() for a, b in zip(before, learner.net.parameters()))
    assert changed                                   # gradients reached the network
    probs = learner.policy().action_probs(ep.trajectories['first_0'][0].obs, np.ones(6, bool))
    assert abs(probs.sum() - 1) < 1e-9
    if kind == 'dqn':
        assert learner._buf['obs'].dtype == np.uint8


@pytest.mark.parametrize('generator', ['hyper', 'conditional'])
def test_gems_on_images(tmp_path, generator, monkeypatch):
    """GEMS end to end on the fake Atari game (registered as an ad-hoc preset)."""
    import config
    import main
    monkeypatch.setitem(config.ENV_PRESETS, 'fake_atari', {'backend': 'test', 'id': 'fake'})
    monkeypatch.setattr(main, 'make_env', lambda preset, **kw: AtariPreprocessing(
        FakeAtari(length=12), max_episode_steps=kw.get('max_episode_steps')))
    main.main(['--method', 'gems', '--env', 'fake_atari', '--budget', '10000000',
               '--results-dir', str(tmp_path), '--gems-generator', generator,
               '--gems-iterations', '1', '--gems-mc-opponents', '1', '--gems-mc-rollouts', '1',
               '--gems-oracle-opponents', '1', '--gems-oracle-rollouts', '2',
               '--gems-mutation-pool', '1', '--gems-random-pool', '1', '--gems-abr-steps', '1',
               '--gems-abr-batch-anchors', '2', '--gems-hyper-hidden', '8',
               '--gems-policy-hidden', '8', '--eval-episodes', '1'])
    assert (tmp_path / 'check' / 'fake_atari' / 'gems' / 'seed0' / 'eval.jsonl').exists()


def _has_space_invaders():
    try:
        from pettingzoo.atari import space_invaders_v2
        space_invaders_v2.parallel_env().reset(seed=0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_space_invaders(), reason='Atari ROMs not installed')
def test_space_invaders_preset():
    env = make_env(parse_env('space_invaders'), max_episode_steps=50)
    assert env.spec.obs_shape == {'first_0': (4, 84, 84), 'second_0': (4, 84, 84)}
    assert env.spec.num_actions == {'first_0': 6, 'second_0': 6}
    ep = play_episode(env, {a: RandomPolicy() for a in env.spec.agents}, np.random.default_rng(0),
                      record=True)
    assert ep.length == 50 and ep.truncated
    assert ep.trajectories['first_0'][0].obs.dtype == np.uint8
