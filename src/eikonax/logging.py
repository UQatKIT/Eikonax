import logging
import sys
from dataclasses import dataclass
from pathlib import Path


# ==================================================================================================
@dataclass
class LoggerSettings:
    log_to_console: bool
    logfile_path: Path | None


@dataclass
class LogValue:
    str_id: str
    str_format: str
    value: int | float | None = None


# ==================================================================================================
class Logger:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        logger_settings: LoggerSettings,
    ) -> None:
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
    def log(self, log_values: dict[str, LogValue]) -> None:
        output_str = ""
        for log_value in log_values.values():
            value_str = f"{log_value.value:{log_value.str_format}}"
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def header(self, log_values: dict[str, LogValue]) -> None:
        log_header_str = ""
        for log_value in log_values.values():
            log_header_str += f"{log_value.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)
