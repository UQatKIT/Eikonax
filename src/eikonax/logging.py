"""Custom logging, based on Python's built-in logger.

!!! info
    Logging can only be used for non-jitted solver loops, it is thus only recommended for eploratory
    runs on small test problems

Classes:
    LoggerSettings: Data class for logger settings
    LogValue: Data class for log values
    Logger: Custom logger class
"""

import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path


# ==================================================================================================
@dataclass
class LoggerSettings:
    """Logger Settings.

    Args:
        log_to_console (bool): If True, log messages will be printed to the console.
        logfile_path (Path): Path to the logfile. If None, no logfile will be generated
    """

    log_to_console: bool
    logfile_path: Path | None


@dataclass
class LogValue:
    """Data class holding info to log.

    Args:
        str_id (str): String identifier to be shown at the top of the log table
        str_format (str): String format for the log value in log file and on consoles
        value (Real): Actual value to log
    """

    str_id: str
    str_format: str
    value: Real | None = None


# ==================================================================================================
class Logger:
    """Custom logger class.

    This class is a minimal wrapper around the Python logger class. It provides handles for logging
    to the console or a file, depending on the user settings. The class's main interface is the
    [`log`][eikonax.logging.Logger.log] method, which takes a list of
    [`LogValue`][eikonax.logging.LogValue] objects and logs them with the given values and string
    formats.

    Methods:
        log: Log the values in the log_values list
        header: Log the header of the log table
        info: Log a message to the console and/or logfile
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        logger_settings: LoggerSettings,
    ) -> None:
        """Constructor of the Logger.

        Sets up log file handlers for printing to console and to file, depending on the user
        settings.

        Args:
            logger_settings (LoggerSettings): Settings for initialization, see `LoggerSettings`
        """
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.log_to_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if logger_settings.logfile_path is not None:
                logger_settings.logfile_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(
                    self._logfile_path,
                    mode="w",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                self._pylogger.addHandler(file_handler)

    # ----------------------------------------------------------------------------------------------
    def log(self, log_values: Iterable[LogValue]) -> None:
        """Log statistics with given values to initialized file handlers.

        Args:
            log_values (Iterable[LogValue]): List of log values
        """
        output_str = ""
        for log_value in log_values:
            value_str = f"{log_value.value:{log_value.str_format}}"
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def header(self, log_values: Iterable[LogValue]) -> None:
        """Log the table header to console and file.

        This method should be invoked once at the beginning of the logging process.

        Args:
            log_values (Iterable[LogValue]): List of log values
        """
        log_header_str = ""
        for log_value in log_values:
            log_header_str += f"{log_value.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        """Wrapper to Python logger `info` call.

        Args:
            message (str): Message to log
        """
        self._pylogger.info(message)
