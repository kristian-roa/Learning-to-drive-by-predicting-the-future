"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
import random
import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape, circleShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car, ENGINE_POWER
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
# TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
TRACK_RAD = 500 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 1600 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.3  # Camera zoom
FIXED_ZOOM = 3
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE * 0.8
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 80 / SCALE * 0.8
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

OBSTACLE_SIZE = 18

ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        status = "None"
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
            status = "tile"
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
            status = "tile"
        if u1 and "obst_friction" in u1.__dict__:
            tile = u1
            obj = u2
            status = "obstacle"
        if u2 and "obst_friction" in u2.__dict__:
            tile = u2
            obj = u1
            status = "obstacle"
        if u1 and "main_car_hull" in u1.__dict__ and u2 and "main_car_hull" in u2.__dict__:
            tile = u1 if u2.main_car_hull else u2
            obj = u2 if u2.main_car_hull else u1
            status = "car"
        if not tile or not obj:
            return
        if "main_car_w" in obj.__dict__ and not obj.main_car_w:
            # blue cars
            if status == 'tile':
                if begin:
                    obj.hull.road_contacts += 1
                else:
                    obj.hull.road_contacts -= 1
            return

        if status == "tile":
            if not obj or "tiles" not in obj.__dict__:
                return

            # calculate distance to the border

            if begin:
                self.env.contacts += 1
                obj.tiles.add(tile)

                if not tile.road_visited:
                    tile.color[0], tile.color[1], tile.color[2] = ROAD_COLOR[0], ROAD_COLOR[1], ROAD_COLOR[2]
                    tile.partner.color[0], tile.partner.color[1], tile.partner.color[2] = ROAD_COLOR[0], ROAD_COLOR[1], ROAD_COLOR[2]

                    tile.road_visited = True
                    tile.partner.road_visited = True
                    # base_r = 1000.0 / len(self.env.track)  # balanced around full track
                    base_r = 3
                    # self.env.reward += base_r if tile.right else base_r / 2  # 1/2 reward if wrong lane
                    self.env.reward += base_r
                    self.env.tile_visited_count += 1
            else:
                self.env.contacts -= 1
                obj.tiles.remove(tile)
        elif status == "obstacle":
            if begin:
                # crashed with obstacle
                self.env.crashed = True

                # tile -> the obstacle
                if not tile.visited:
                    self.env.reward -= 30  # punishment for crashing with obstacle
                    self.env.crash_count += 0.5
                tile.visited = True
                if tile.partner is not None: tile.partner.visited = True
        elif status == "car":
            if begin:
                self.env.crashed = True
                # tile -> the other car
                if not tile.crashed:
                    self.env.reward -= 30
                    self.env.crash_count += 1
                tile.crashed = True


class CarRacing(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1, scenario=0, obstacles=3, seed=None, train=True):
        EzPickle.__init__(self)
        self.seed(seed)
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.circle = fixtureDef(shape=circleShape(pos=(0, 0), radius=OBSTACLE_SIZE / SCALE))

        self.action_space = spaces.Box(
            np.array([0, 0, 0, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1, +1, +1]).astype(np.float32),
        )  # NO-OP, steer, gas, brake -> NO-OP, left, right, gas, brake

        self.actions_discrete_names = ['NO-OP', 'LEFT', 'RIGHT', 'GAS', 'BRAKE']
        self.actions_discrete = [np.array([0.0, 0.0, 0.0]),
                                 np.array([-1.0, 0.0, 0.0]),
                                 np.array([1.0, 0.0, 0.0]),
                                 np.array([0.0, 1.0, 0.0]),
                                 np.array([0.0, 0.0, 0.8])]

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.steps = 0
        self.scenario = scenario
        self.obstacle_count = obstacles
        self.train = train

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road + self.obstacles + self.obstacles_sq:
            self.world.DestroyBody(t)
        self.road = []
        self.obstacles = []
        self.obstacles_sq = []
        self.car.destroy()
        for c in self.crossing_cars: c.destroy()

    def _create_track(self):
        CHECKPOINTS = 12
        DIRECTION = random.sample((-1, 1), 1)[0]

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 2, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, (prev_beta * 0.5 + beta * 0.5)*DIRECTION, x*DIRECTION, y, True))  # mirrors track if RIGHT
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                    track[i][0] > self.start_alpha >= track[i - 1][0]
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1: i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        self._build_road(track)

        self.track = track
        return True

    def _create_scenario(self, scenario=1):
        DIRECTION = 1

        # Create checkpoints
        checkpoints = []
        if scenario == 1:
            self.OBS_ORDER = random.sample((-1, 1), 1)[0]

            # (x, y)
            checkpoints = [(0, c, False) for c in range(-4, 1)]
            checkpoints += [(0, c) for c in range(1, 51)]
            checkpoints += [(0, c, False) for c in range(51, 65)]
        elif scenario == 2:
            DIRECTION = random.sample((-1, 1), 1)[0]
            rad = 10

            checkpoints = [(0, c, False) for c in range(-4, 1)]
            checkpoints += [(0, c) for c in range(1, 13)]  # straight up

            alpha = 0
            checkpoints += [(rad*math.cos(alpha := alpha + np.deg2rad(10)) - 10, 12 + rad*math.sin(alpha)) for _ in range(9)]  # left turn

            # (-10, 22)
            checkpoints += [(c, 22) for c in range(-11, -20, -1)]  # straight left

            # (19, 22)
            checkpoints += [(rad*math.cos(alpha := alpha + np.deg2rad(10)) - 18, 32-rad*math.sin(alpha)) for _ in range(9)]  # right turn

            # (-28, 32)
            checkpoints += [(-28, c) for c in range(32, 45)]  # straight up

            checkpoints += [(-28, c, False) for c in range(45, 59)]
        elif scenario == 3:
            checkpoints = [(0, c, False) for c in range(-4, 1)]
            checkpoints += [(0, c) for c in range(1, 31)]
            checkpoints += [(0, c, False) for c in range(31, 46)]

            mid_border = [(np.deg2rad(90.0), x*TRACK_DETAIL_STEP, 15*TRACK_DETAIL_STEP) for x in range(14, -15, -1)]
            track2 = [(np.deg2rad(90.0), x*TRACK_DETAIL_STEP, 15*TRACK_DETAIL_STEP)
                      for x in (14, 4.28, -3.80, -14)]
        elif scenario == 4:
            DIRECTION = random.sample((-1, 1), 1)[0]
            self.OBS_ORDER = random.sample((-1, 1), 1)[0]
            self.TURNS_FIRST = random.randint(0, 1)
            rad = 10
            alpha = 0

            # (x, y)
            checkpoints = [(0, 0.5)]
            checkpoints += [(0, c) for c in range(2, 52)]  # straight up

            # (0, 51)
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 9.8, 51 + rad * math.sin(alpha)) for _ in range(9)]  # left turn

            # (-9.8, 61)
            checkpoints += [(c, 61) for c in range(-11, -40, -1)]  # straight left

            # (-39, 61)
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 38.3, 51.2 + rad * math.sin(alpha)) for _ in range(9)]  # left turn

            # (-48.3, 51.2)
            checkpoints += [(-48.3, c) for c in range(50, 36, -1)]  # straight down

            # (-48.3 , 37)
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 38.4, 37.8 + rad * math.sin(alpha)) for _ in range(9)]  # left turn

            # (-38.4, 27.8)
            checkpoints += [(c, 27.8) for c in range(-37, -28)]  # straight right

            # (-29, 27.8)
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 30, 17.94 - rad * math.sin(alpha)) for _ in range(9)]  # right turn
            alpha -= np.deg2rad(180)

            # (-20, 17.94)
            checkpoints += [(-20, c) for c in range(17, 0, -1)]  # straight down

            # (-20, 1) -> (0, 1)
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 10.16, 1.0 + rad * math.sin(alpha)) for _ in range(9)]  # last turn
            checkpoints += [(rad * math.cos(alpha := alpha + np.deg2rad(10)) - 10, 1.0 + rad * math.sin(alpha)) for _ in range(9)]

        self.road = []

        # Go from one checkpoint to another to create track
        track = []
        prev_x, prev_y = (0, -5) if self.scenario in (1, 2, 3) else (0, 0)
        # for x, y, is_road in checkpoints:
        for cp in checkpoints:
            (x, y, is_road) = (*cp, True) if len(cp) == 2 else cp

            beta = np.arctan2(y - prev_y, x - prev_x) - np.pi / 2
            prev_x, prev_y = x, y
            track.append((0, beta * DIRECTION, x*TRACK_DETAIL_STEP * DIRECTION, y*TRACK_DETAIL_STEP, is_road))  # mirrors track if RIGHT

        self._build_road(track, closed_loop=False)

        # create crossing road for scenario 3
        if scenario == 3:
            # Create tiles - right + left track
            for (beta1, x1, y1), (beta2, x2, y2) in zip(track2[::2], track2[1:][::2]):
                road1_l = (
                    x1 - TRACK_WIDTH * math.cos(beta1),
                    y1 - TRACK_WIDTH * math.sin(beta1),
                )
                road1_r = (
                    x1 + (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                road2_l = (
                    x2 - TRACK_WIDTH * math.cos(beta2),
                    y2 - TRACK_WIDTH * math.sin(beta2),
                )
                road2_r = (
                    x2 + (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                v = [road1_l, road1_r, road2_r, road2_l]

                self.fd_tile.shape.vertices = v
                t = self.world.CreateStaticBody(fixtures=self.fd_tile)
                t.userData = t
                t.color = ROAD_COLOR
                t.road_friction = 1.0
                t.road_visited = True
                t.fixtures[0].sensor = True
                t.right = True
                t.partner = t

                self.road_poly.append((v, t.color))
                self.road.append(t)

                # outer border
                b1_l = (
                    x1 + (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b1_r = (
                    x1 + (TRACK_WIDTH + 2 * BORDER) * math.cos(beta1),
                    y1 + (TRACK_WIDTH + 2 * BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                b2_r = (
                    x2 + (TRACK_WIDTH + 2 * BORDER) * math.cos(beta2),
                    y2 + (TRACK_WIDTH + 2 * BORDER) * math.sin(beta2),
                )
                self.road_poly.insert(0, ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1)))

                # inside border
                b1_l = (
                    x1 - (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 - (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b1_r = (
                    x1 - TRACK_WIDTH * math.cos(beta1),
                    y1 - TRACK_WIDTH * math.sin(beta1),
                )
                b2_l = (
                    x2 - (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 - (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                b2_r = (
                    x2 - TRACK_WIDTH * math.cos(beta2),
                    y2 - TRACK_WIDTH * math.sin(beta2),
                )
                self.road_poly.insert(0, ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1)))

            # middle border
            for (beta1, x1, y1), (beta2, x2, y2) in zip(mid_border[1:][::2], mid_border[2:][::2]):
                b1_l = (x1, y1,)
                b1_r = (x1 + BORDER * math.cos(beta1), y1 + BORDER * math.sin(beta1),)
                b2_l = (x2, y2,)
                b2_r = (x2 + BORDER * math.cos(beta2), y2 + BORDER * math.sin(beta2),)
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (0.8, 0.8, 0)))

            # end borders of crossing road
            border = [track2[0], track2[-1]]
            for i, (beta, x, y) in enumerate(border):
                road1_l = (
                    x - (TRACK_WIDTH + BORDER) * math.cos(beta),
                    y - (TRACK_WIDTH + BORDER) * math.sin(beta),
                )
                road1_r = (
                    x + (TRACK_WIDTH + BORDER) * math.cos(beta),
                    y + (TRACK_WIDTH + 2*BORDER) * math.sin(beta),
                )
                road2_l = (
                    x + (BORDER if i == 0 else -BORDER) - (TRACK_WIDTH + BORDER) * math.cos(beta),
                    y - (TRACK_WIDTH + BORDER) * math.sin(beta),
                )
                road2_r = (
                    x + (BORDER if i == 0 else -BORDER) + (TRACK_WIDTH + BORDER) * math.cos(beta),
                    y + (TRACK_WIDTH + 2*BORDER) * math.sin(beta),
                )
                vertices = [road1_l, road1_r, road2_r, road2_l]
                self.road_poly.insert(0, (vertices, (1, 1, 1)))

        self.track = track
        return True

    def _build_road(self, track, closed_loop=True):
        self.track_length = 0

        # Create tiles - right + left track
        for i in range(0 if closed_loop else 1, len(track)):
            _, beta1, x1, y1, is_road = track[i]
            _, beta2, x2, y2, _ = track[i - 1]
            road1_l = (
                x1,
                y1,
            )
            road1_r = (
                x1 + (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            road2_l = (
                x2,
                y2,
            )
            road2_r = (
                x2 + (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            vertices_r = [road1_l, road1_r, road2_r, road2_l]

            road3_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road3_r = (
                x1,
                y1,
            )
            road4_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road4_r = (
                x2,
                y2,
            )
            vertices_l = [road3_l, road3_r, road4_r, road4_l]

            c = 0.01 * (i % 3)
            color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]

            vertices = [vertices_r, vertices_l]
            for idx, v in enumerate(vertices):
                if is_road:
                    self.fd_tile.shape.vertices = v
                    t = self.world.CreateStaticBody(fixtures=self.fd_tile)
                    t.userData = t
                    t.color = color
                    t.road_visited = False
                    t.right = not idx
                    t.road_friction = 1.0
                    t.fixtures[0].sensor = True

                    # left and right part of road gets a pointer to each other
                    # don't want points for visiting both left and right part of same road tile
                    if idx == 0:
                        right = t
                        self.track_length += 1
                    else:
                        t.partner = right
                        right.partner = t

                    self.road.append(t)
                self.road_poly.append((v, color))

            # middle border
            if i % 2 == 0:
                b1_l = (x1, y1,)
                b1_r = (x1 + BORDER * math.cos(beta1), y1 + BORDER * math.sin(beta1),)
                b2_l = (x2, y2,)
                b2_r = (x2 + BORDER * math.cos(beta2), y2 + BORDER * math.sin(beta2),)
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (0.8, 0.8, 0)))

            # outer border
            b1_l = (
                x1 + (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b1_r = (
                x1 + (TRACK_WIDTH + 2 * BORDER) * math.cos(beta1),
                y1 + (TRACK_WIDTH + 2 * BORDER) * math.sin(beta1),
            )
            b2_l = (
                x2 + (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            b2_r = (
                x2 + (TRACK_WIDTH + 2 * BORDER) * math.cos(beta2),
                y2 + (TRACK_WIDTH + 2 * BORDER) * math.sin(beta2),
            )
            self.road_poly.insert(0, ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1)))

            # inside border
            b1_l = (
                x1 - (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 - (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b1_r = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            b2_l = (
                x2 - (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 - (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            b2_r = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            self.road_poly.insert(0, ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1)))

    def _create_obstacles(self, filled_obstacles=False):
        self.obstacles = []
        self.obstacles_sq = []
        off1 = TRACK_WIDTH / 2 + BORDER + BORDER / 2
        off2 = -TRACK_WIDTH / 2 - BORDER / 2

        if self.scenario == 0:
            # pick 3 random tiles with some space in between
            # do not pick tiles where the car spawns

            # to avoid obstacles in the middle of a turn
            candidates = [self.track[15:-15][i] for i in range(len(self.track[15:-15]))
                          if abs(self.track[15:-15][(i+3) % len(self.track[15:-15])][1] - self.track[15:-15][i-3][1]) < 0.25]

            tiles = [x[1:-1] for x in random.sample(candidates[::20], k=self.obstacle_count)]

        elif self.scenario == 1:
            if self.obstacle_count > 0:
                tiles = [self.track[20][1:-1], self.track[40][1:-1]]
            else:
                tiles = []
        elif self.scenario == 4 and self.obstacle_count != 0:
            tiles = [self.track[15][1:-1], self.track[35][1:-1], self.track[74][1:-1]]
        else:
            tiles = []

        for i, (angle, x, y) in enumerate(tiles):
            if self.scenario == 0:
                offset = random.uniform(BORDER, TRACK_WIDTH + BORDER)
            elif self.scenario in (1, 4):
                if i % 2 != np.clip(self.OBS_ORDER, 0, 1):
                    offset = off1
                else:
                    offset = off2
                if self.scenario == 4 and i == len(tiles) - 1: offset = BORDER / 2

            self.circle.shape.pos = (x + offset * math.cos(angle), y + offset * math.sin(angle))
            o = self.world.CreateStaticBody(fixtures=self.circle)
            o.userData = o
            o.color = (0.55, 0, 0)
            o.obst_friction = 1.0
            o.visited = False
            o.partner = None  # for obstacle fill

            self.obstacles.append(o)

        # for scenario 1 and 4
        if filled_obstacles:
            for i, obstacle in enumerate(self.obstacles[:2]):
                pos = obstacle.fixtures[0].shape.pos
                radius = obstacle.fixtures[0].shape.radius

                x = 4.6
                right = i % 2 != np.clip(self.OBS_ORDER, 0, 1)

                vertice_corners = [(0, -radius), (x, -radius), (0, radius), (x, radius)] if right else \
                                  [(-x, -radius), (0, -radius), (-x, radius), (0, radius)]

                # vertices = (bot left, bot right, top right, top left)
                self.fd_tile.shape.vertices = [pos + v for v in vertice_corners]
                o = self.world.CreateStaticBody(fixtures=self.fd_tile)
                o.userData = o
                o.color = (0.55, 0, 0)
                o.obst_friction = 1.0
                o.visited = False
                o.partner = obstacle

                self.obstacles_sq.append(o)

    def distance_to_obstacle(self):
        # calculates shortest distance from all 4 wheels to all obstacles
        wheels = [w.position for w in self.car.wheels]

        # give distance if it's below 50  TODO: Should probably change to 30 as that is within sight
        return min([max(0, np.linalg.norm(w_pos - obst.fixtures[0].shape.pos) - obst.fixtures[0].shape.radius)
                    for obst in self.obstacles for w_pos in wheels] + [50])

    def distance_to_other_cars(self):
        # calculates shortest distance from all 4 wheels to all crossing cars
        wheels = [w.position for w in self.car.wheels]

        # only works for scenario 3
        return min([np.linalg.norm(w_pos - car.hull.position) for car in self.crossing_cars for w_pos in wheels] + [50])

    def reset(self):
        self._destroy()
        self.steps = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.crash_count = 0
        self.t = 0.0
        self.road_poly = []
        self.crossing_cars = []

        # stuff for determining car on road or not
        self.contacts = 0

        self.last_action = []
        self.crashed = False

        while True:
            success = self._create_track() if self.scenario == 0 else self._create_scenario(self.scenario)
            if success:
                # self._create_obstacles(filled_obstacles=self.scenario in (1, 4))
                self._create_obstacles()
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        i = 1
        while not self.track[i][-1]: i += 1

        c_angle, c_x, c_y = (0, *self.track[154][2:4]) if self.scenario == 4 and self.TURNS_FIRST else self.track[i][1:4]
        c_x += -TRACK_WIDTH / 2 if (self.scenario == 1 or self.scenario == 4 and not self.TURNS_FIRST)\
                                   and self.OBS_ORDER == -1 else (TRACK_WIDTH / 2 + BORDER) * math.cos(c_angle)

        self.car = Car(self.world, c_angle, c_x, c_y + (TRACK_WIDTH / 2 + BORDER) * math.sin(c_angle))

        st = self.step(None)
        self.measurement_size = len(st[2])
        return st[0]

    def step(self, action):
        if action is not None:
            self.last_action = []
            if action[1]:
                self.last_action.append('GAS')
            if action[2]:
                self.last_action.append('BRAKE')
            if action[0]:
                self.last_action.append('LEFT' if action[0] < 0 else 'RIGHT')
            if not any(action):
                self.last_action = ['NO-OP']

            sp = np.linalg.norm(self.car.hull.linearVelocity)

            restrict_speed = True

            speed_value = 0 if restrict_speed and sp >= 30 else action[1]

            self.car.steer(-action[0] * 0.3)  # 0.4 is default
            self.car.gas(speed_value)
            self.car.brake(action[2])

            # accelerate very slightly if the car turns while standing still
            if action[0] != 0 and sp < 0.1: self.car.gas(1)

        # spawn blue car
        if self.scenario == 3 and (round(self.t * FPS) - 40) % 110 == 0:
            pos = (13, 17)
            c = Car(self.world, np.pi/2 + 0.04, pos[0]*TRACK_DETAIL_STEP, pos[1]*TRACK_DETAIL_STEP, main_car=False)
            c.hull.color = (0.0, 0.0, 0.8)
            c.hull.linearVelocity = (random.randint(-50, 0), 0)
            self.crossing_cars.append(c)

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        if self.scenario == 3:
            for c in self.crossing_cars:
                if random.randint(1, 10) >= 7: c.gas(1)
                c.step(1.0 / FPS)

                if c.hull.road_contacts < 4:
                    c.destroy()
                    self.crossing_cars.remove(c)

        self.state = self.render("state_pixels")
        distance_to_obstacle = self.distance_to_other_cars() if self.scenario == 3 else self.distance_to_obstacle()
        speed = np.linalg.norm(self.car.hull.linearVelocity)

        proximity_penalty = round(((distance_to_obstacle if distance_to_obstacle <= 10 else 10) - 10) / 50, 2)

        step_reward = 0
        done = False

        if self.crashed: done = True

        if action is not None:  # First step without action, called from reset()
            self.steps += 1
            self.reward -= 0.1
            # self.reward += proximity_penalty

            # fuel spent per step, accumulate if I want to give this to DFP !!!
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -= 10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # track completed
            if self.tile_visited_count == self.track_length:
                done = True
                step_reward += 30  # bonus for completion

            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        # outside of border
        if self.train and self.contacts == 0:
            done = True
            step_reward = -30
            self.crash_count += 1

        measurements = (distance_to_obstacle,
                        speed,
                        self.tile_visited_count,
                        self.crash_count)

        return self.state, step_reward, measurements, done, {}

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

            self.extra_label = pyglet.text.Label(
                "0000",
                font_size=18,
                x=20,
                y=WINDOW_H * 4.2 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )

        if "t" not in self.__dict__:
            return  # reset() not called yet

        if ZOOM_FOLLOW:
            # Animate zoom first second:
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
            scroll_x = self.car.hull.position[0]
            scroll_y = self.car.hull.position[1]
            angle = -self.car.hull.angle
            vel = self.car.hull.linearVelocity
            if np.linalg.norm(vel) > 0.5:
                angle = math.atan2(vel[0], vel[1])
            self.transform.set_scale(zoom, zoom)
            self.transform.set_translation(
                WINDOW_W / 2
                - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
                WINDOW_H / 4
                - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
            )
            self.transform.set_rotation(angle)
        else:
            self.transform.set_scale(FIXED_ZOOM, FIXED_ZOOM)
            self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 2) if self.scenario == 0 else self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 3.5)

        self.car.draw(self.viewer, mode != "state_pixels")

        if self.scenario == 3:
            for c in self.crossing_cars:
                c.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        self.render_obstacles()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_obstacles(self):
        for obstacle in self.obstacles_sq:
            for f in obstacle.fixtures:
                self.viewer.draw_polygon(f.shape.vertices, color=obstacle.color)
                self.viewer.draw_polygon(f.shape.vertices, color=(0, 0, 0), filled=False, linewidth=2)

        for obstacle in self.obstacles:
            for f in obstacle.fixtures:
                trans = f.body.transform
                t = rendering.Transform(translation=trans * f.shape.pos)
                self.viewer.draw_circle(f.shape.radius, 20, color=obstacle.color).add_attr(t)
                self.viewer.draw_circle(f.shape.radius, 20, color=(0, 0, 0), filled=False, linewidth=2).add_attr(t)

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

        self.score_label.text = f"{int(self.reward):04d}"
        self.score_label.draw()

        self.extra_label.text = f"SPEED: {int(true_speed):02d} ACTION: {' + '.join(self.last_action)}"
        self.extra_label.draw()


if __name__ == "__main__":
    from pyglet.window import key

    seed = 73

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == key.R:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0


    scenario = int(sys.argv[1])
    obstacles = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    env = CarRacing(scenario=scenario, obstacles=obstacles, seed=seed, verbose=0)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    record_video = False
    if record_video:
        from gym.wrappers.record_video import RecordVideo
        env = RecordVideo(env, "../../../../Recordings/Manual_control",
                          name_prefix=f'scenario{scenario}{"_" + str(obstacles) if scenario == 0 else ""}')

    isopen = True
    GAME = 0
    while isopen:
        env.reset()
        total_reward = 0.0
        GAME += 1
        restart = False
        while True:
            s, r, sensors, done, info = env.step(a)
            total_reward += r
            '''
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            '''

            if done:
                dist_to_obst, speed, tile_visited_count, _ = sensors
                print("GAME", GAME,
                      "/ STEPS", env.steps,
                      "/ TILES", f"{tile_visited_count}/{env.track_length}",
                      "/ REWARD", round(total_reward))

            isopen = env.render()
            if done or restart or not isopen:
                break
    env.close()
