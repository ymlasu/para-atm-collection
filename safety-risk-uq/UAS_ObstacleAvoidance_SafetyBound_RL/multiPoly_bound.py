import numpy as np
import sys
from gym.envs.toy_text import discrete
from boundSize import geometry

UP = 3
RIGHT = 0
DOWN = 1
LEFT = 2
g=geometry()

polysize1 = [5, 15, 10, 20]
polysize2 = [30, 45, 15, 25]
polysize3 = [10, 15, 28, 40]
polysize4 = [25, 30, 35, 49]

def polyReal(polysize):
    return [(polysize[0]*4, polysize[2]*4), (4*polysize[0], 4*(polysize[3]+1)), (4*(polysize[1]+1), 4*(polysize[3]+1)), (4*(polysize[1]+1), 4*polysize[2])]

poly1 = polyReal(polysize1)
poly2 = polyReal(polysize2)
poly3 = polyReal(polysize3)
poly4 = polyReal(polysize4)

def polyFlag(g, poly1, poly2, poly3, poly4, boundp, c1, c2, r):
    flag =0

    if (
            g.Flagrectc(poly1, c1, r) == 1 or g.Flagrectc(poly1, c2, r) == 1 or g.Flag2rect(poly1, boundp) == 1 or
            g.Flagrectc(poly2, c1, r) == 1 or g.Flagrectc(poly2, c2, r) == 1 or g.Flag2rect(poly2, boundp) == 1 or
            g.Flagrectc(poly3, c1, r) == 1 or g.Flagrectc(poly3, c2, r) == 1 or g.Flag2rect(poly3, boundp) == 1 or
            g.Flagrectc(poly4, c1, r) == 1 or g.Flagrectc(poly4, c2, r) == 1 or g.Flag2rect(poly4, boundp) == 1
    ):
        flag = 1

    return flag

class CliffWalkingEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, origin, des):
        self.shape = (50, 50)
        self.des= des
        nS = np.prod(self.shape)
        nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[polysize1[0]:polysize1[1]+1, polysize1[2]:polysize1[3]+1] = True
        self._cliff[polysize2[0]:polysize2[1]+1, polysize2[2]:polysize2[3]+1] = True
        self._cliff[polysize3[0]:polysize3[1]+1, polysize3[2]:polysize3[3]+1] = True
        self._cliff[polysize4[0]:polysize4[1]+1, polysize4[2]:polysize4[3]+1] = True

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-5, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 4])
            P[s][DOWN] = self._calculate_transition_prob(position, [3, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -4])

        isd = np.zeros(nS)
        isd[np.ravel_multi_index(origin, self.shape)] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord


    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        reward = -0.5 * np.sqrt((new_position[0] - self.des[0]) ** 2 + (new_position[1] - self.des[1]) ** 2)
        # up
        if delta == [-5, 0]:
            lh = 66.3
            r = 1.6
            c1 = (current[0]*4+2, current[1]*4+2)
            c2 = (c1[0]-lh, c1[1])
            boundp = [(c1[0], c1[1]-r), (c2[0], c2[1]-r), (c2[0], c2[1]+r),(c1[0], c1[1]+r)]

            ff = polyFlag(g, poly1, poly2, poly3, poly4, boundp, c1, c2, r)
            reward = reward - 1000 * ff
        # down
        if delta == [3, 0]:
            lh = 30.8
            r = 1.6
            c1 = (current[0] * 4 + 2, current[1] * 4 + 2)
            c2 = (c1[0] + lh, c1[1])
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]

            ff = polyFlag(g, poly1, poly2, poly3, poly4, boundp, c1, c2, r)
            reward = reward - 1000 * ff
        # right
        if delta == [0, 4]:
            lh = 46.9
            r = 7.1
            c1 = (current[0] * 4 + 2, current[1] * 4 + 2)
            c2 = (c1[0], c1[1] + lh)
            boundp = [(c1[0] - r, c1[1]), (c2[0] - r, c2[1]), (c2[0]+r, c2[1]), (c1[0]+r, c1[1])]

            ff = polyFlag(g, poly1, poly2, poly3, poly4, boundp, c1, c2, r)
            reward = reward - 1000 * ff
        # left
        if delta == [0, -4]:
            lh = 47.3
            r = 7.1
            c1 = (current[0] * 4 + 2, current[1] * 4 + 2)
            c2 = (c1[0], c1[1] - lh)
            boundp = [(c1[0] - r, c1[1]), (c2[0] - r, c2[1]), (c2[0] + r, c2[1]), (c1[0] + r, c1[1])]

            ff = polyFlag(g, poly1, poly2, poly3, poly4, boundp, c1, c2, r)
            reward = reward - 1000 * ff

        if self._cliff[tuple(new_position)]:
            reward = reward -1000

        if (new_position == self.des).all():
            reward = 1

        is_done = False
        if tuple(new_position) == self.des or np.sqrt((new_position[0]- self.des[0])**2 + (new_position[1]- self.des[1])**2) <4:
            is_done = True

        return [(1.0, new_state, reward, is_done)]

