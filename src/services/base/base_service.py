from ds_util import LoggerUtil


class BaseService:
    def __init__(self):
        self.logger = LoggerUtil.get_logger(self.__class__.__name__)
        self.logger.info(f"Setup [{self.__class__.__name__}]")
