# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia  
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-28 13:43:50
# @Last Modified time: 2016-04-28 13:44:59
# @FileName: mylogger.py

"""logger 生成器.
"""

import os
import logging
import logging.handlers


class Logger(object):
    """logger 生成器"""
    @staticmethod
    def get_logger(service='', level=logging.DEBUG,
                   level_stream=logging.DEBUG, level_trfile=logging.INFO):
        """需要确保 service 唯一"""
        logger = logging.getLogger(service)
        logger.setLevel(level)

        log_fname = '{0}.log'.format(service) if service else 'logger.log'

        handlers = Logger.get_handlers(fname=log_fname,
                                       level_stream=level_stream,
                                       level_trfile=level_trfile)
        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @staticmethod
    def get_formatter(service=''):
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(levelname)s][%(filename)s][%(threadName)s][%(funcName)s][%(lineno)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        return formatter

    @staticmethod
    def get_handlers(fname='logger.log', level_stream=None,
                     level_trfile=None, **kwargs):
        """提供两类 handler，一类在终端打印，一类写入到文件中.

        参数:
        + fname: 日志写入的文件名，默认 'logger.log'
        + level_stream: stream logging 的级别，默认 DEBUG
        + level_trfile: time rotating filie logging 的级别，默认 WARNING
        + when_trfile: 日志滚动形式，默认 'W0'
        + encoding: 日志写入的编码，默认 'utf-8'

        XXX:
          + 参数检验
        """
        fname = fname or 'logger.log'
        level_stream = level_stream if level_stream else logging.DEBUG
        level_trfile = level_trfile if level_trfile else logging.INFO
        when_trfile = kwargs.get('when_trfile', 'W0')
        encoding = kwargs.get('encoding', 'utf-8')


    
        absdir = os.path.dirname(os.path.abspath(__file__))
        dir_log = os.path.join(absdir, 'log')
        if not os.path.exists(dir_log):
            os.makedirs(dir_log)

        logfile_warning = os.path.join(dir_log, fname)

        formatter = Logger.get_formatter()

        # streaming handler
        handler_stream = logging.StreamHandler()
        handler_stream.setLevel(level_stream)
        handler_stream.setFormatter(formatter)

        # timed rotating file handler for waring+
        handler_tf_warn = logging.handlers.TimedRotatingFileHandler(logfile_warning,
                                                                    when=when_trfile,
                                                                    encoding=encoding)
        handler_tf_warn.setLevel(level_trfile)
        handler_tf_warn.setFormatter(formatter)

        return [handler_stream, handler_tf_warn]
