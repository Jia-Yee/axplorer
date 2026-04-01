from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment
from src.envs.hypercube_diameter import HypercubeDiameterEnvironment

ENVS = {
    "square": SquareEnvironment, 
    "isosceles": IsoscelesEnvironment, 
    "sphere": SphereEnvironment,
    "hypercube": HypercubeDiameterEnvironment,  # New environment for hypercube diameter problem
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
