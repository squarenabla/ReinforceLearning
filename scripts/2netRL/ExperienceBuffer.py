import threading
import random

class ExperienceBuffer:
    def __init__(self, buffer_size=5000):
        self.buffer_lock = threading.Lock()

        self.buffer_size = buffer_size
        self.states = []
        self.actions = []
        self.advantages = []
        self.future_rewards = []

    def add(self, st, act, adv, fr):
        assert (len(st) == len(act) == len(adv) == len(fr)), "sizes are different!"
        self.buffer_lock.acquire()
        self.states.append(st)
        self.actions.append(act)
        self.advantages.append(adv)
        self.future_rewards.append(fr)
        if len(self.states) > self.buffer_size:
            del self.states[:(len(self.states) - self.buffer_size)]
            del self.actions[:(len(self.actions) - self.buffer_size)]
            del self.advantages[:(len(self.advantages) - self.buffer_size)]
            del self.future_rewards[:(len(self.future_rewards) - self.buffer_size)]

        self.buffer_lock.release()

    def sample(self, size):
        self.buffer_lock.acquire()
        indexes = random.sample(range(len(self.states)), min(size, len(self.states)))
        ret_states = [self.states[i] for i in indexes]
        ret_actions = [self.actions[i] for i in indexes]
        ret_advatages = [self.advantages[i] for i in indexes]
        ret_futurer = [self.future_rewards[i] for i in indexes]
        self.buffer_lock.release()

        return ret_states, ret_actions, ret_advatages, ret_futurer

    def getCompasity(self):
        return len(self.states)