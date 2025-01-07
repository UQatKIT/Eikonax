import warnings

from beartype import BeartypeConf, BeartypeStrategy, beartype
from beartype.claw import beartype_this_package

nobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.O0))
beartype_this_package()

warnings.filterwarnings("ignore", message="jax.*", category=DeprecationWarning)
