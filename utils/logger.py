"""日志系统模块"""
import os
import logging
from datetime import datetime


class Logger:
    """日志系统类"""
    
    def __init__(self):
        self.logger = None
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统，同时输出到文件和控制台"""
        # 创建logs目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 日志文件名带时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"magicraft_{timestamp}.log")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
    
    def log(self, message, level='info'):
        """记录日志"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)

