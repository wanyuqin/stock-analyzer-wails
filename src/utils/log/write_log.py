import logging
import logging.handlers
import json
import os
from contextvars import ContextVar
from typing import Optional
from pathlib import Path

from coze_coding_utils.runtime_ctx.context import Context
from utils.log.config import LOG_DIR

request_context: ContextVar[Optional[Context]] = ContextVar('request_context', default=None)


class ContextFilter(logging.Filter):
    
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = request_context.get()
        
        if ctx:
            record.log_id = ctx.logid or ''
            record.run_id = ctx.run_id or ''
            record.space_id = ctx.space_id or ''
            record.project_id = ctx.project_id or ''
            record.method = ctx.method or ''
            record.x_tt_env = ctx.x_tt_env or ''
        else:
            record.log_id = ''
            record.run_id = ''
            record.space_id = ''
            record.project_id = ''
            record.method = ''
            record.x_tt_env = ''
        
        return True


class APSchedulerFilter(logging.Filter):
    
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith('apscheduler'):
            message = record.getMessage()
            if 'Running job' in message or 'next run at:' in message:
                return False
        return True


class JsonFormatter(logging.Formatter):
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'message': record.getMessage(),
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'log_id': getattr(record, 'log_id', ''),
            'run_id': getattr(record, 'run_id', ''),
            'space_id': getattr(record, 'space_id', ''),
            'project_id': getattr(record, 'project_id', ''),
            'method': getattr(record, 'method', ''),
            'x_tt_env': getattr(record, 'x_tt_env', ''),
            'lineno': record.lineno,
            'funcName': record.funcName,
        }

        if record.exc_info:
            log_data['exc_info'] = self.formatException(record.exc_info)
        
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'log_id', 'run_id', 'space_id', 'project_id', 'method',
                          'x_tt_env', 'rpc_persist_rec_rec_biz_scene', 
                          'rpc_persist_coze_record_root_id', 'rpc_persist_rec_root_entity_type',
                          'rpc_persist_rec_root_entity_id']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class PlainTextFormatter(logging.Formatter):
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'message': record.getMessage(),
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'log_id': getattr(record, 'log_id', ''),
            'run_id': getattr(record, 'run_id', ''),
            'space_id': getattr(record, 'space_id', ''),
            'project_id': getattr(record, 'project_id', ''),
            'method': getattr(record, 'method', ''),
            'x_tt_env': getattr(record, 'x_tt_env', ''),
            'lineno': record.lineno,
            'funcName': record.funcName,
        }
        
        if record.exc_info:
            log_data['exc_info'] = self.formatException(record.exc_info)
        
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'log_id', 'run_id', 'space_id', 'project_id', 'method',
                          'x_tt_env', 'rpc_persist_rec_rec_biz_scene', 
                          'rpc_persist_coze_record_root_id', 'rpc_persist_rec_root_entity_type',
                          'rpc_persist_rec_root_entity_id']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_file: Optional[str] = None,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
    log_level: str = "INFO",
    use_json_format: bool = True,
    console_output: bool = True
):
    
    if log_file is None:
        try:
            log_dir = Path(LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / 'app.log')
        except Exception as e:
            fallback_log_dir = Path('/tmp/work/logs/bypass')
            fallback_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(fallback_log_dir / 'app.log')
            print(f"Warning: Using fallback log directory: {fallback_log_dir}, due to error: {e}", flush=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    root_logger.handlers.clear()
    
    context_filter = ContextFilter()
    apscheduler_filter = APSchedulerFilter()
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if use_json_format:
        file_formatter = JsonFormatter()
    else:
        file_formatter = PlainTextFormatter(
            fmt='%(asctime)s %(levelname)s [log_id=%(log_id)s] [run_id=%(run_id)s] %(name)s:%(lineno)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(context_filter)
    file_handler.addFilter(apscheduler_filter)
    root_logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        console_formatter = PlainTextFormatter(
            fmt='%(asctime)s %(levelname)s [log_id=%(log_id)s] [run_id=%(run_id)s] %(name)s:%(lineno)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(context_filter)
        console_handler.addFilter(apscheduler_filter)
        root_logger.addHandler(console_handler)
    
    logging.info(f"Logging configured: file={log_file}, max_bytes={max_bytes}, backup_count={backup_count}")
    
    return log_file


__all__ = ['setup_logging', 'request_context', 'ContextFilter', 'APSchedulerFilter', 'JsonFormatter', 'PlainTextFormatter']
