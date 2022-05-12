try:
    import Box2D
    from gym.envs.box2d.car_racing import CarRacing
except ImportError:
    Box2D = None
