import warnings

from beartype.claw import beartype_this_package

beartype_this_package()

warnings.filterwarnings("ignore", message="jax.*", category=DeprecationWarning)
