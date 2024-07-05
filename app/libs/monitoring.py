from enum import Enum
import sys
import time
import json


class LogLevel(str, Enum):
    
    INFO: str = 'INFO'
    ALERT: str = 'ALERT'
    WARNING: str = 'WARNING'
    DEBUG: str = 'DEBUG'
    ERROR: str = 'ERROR'
    CRITICAL: str = 'CRITICAL'


class Logs():

    def __init__(self):
        self.critical_logs = []

        
    def print_log(
        self,
        level: LogLevel,
        message: str,
        LineNumber: int = 0,
        ErrorDescription: str=''
    ):
        '''Prints log in JSON format with time, level and log attributes.

        Args:
            level (str):  LogLevel.
            message (str): log message string.
            LineNumber (str): application line number where log is printed.
            ErrorDescription (str): string with error description.  
            Email (list): list of emails.
        '''

        log = {}
        try:
            log['time'] = time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', time.localtime()
            )
            log['level'] = level

            if not ErrorDescription:
                if message[0] == '{' and message[-1] == '}':
                    log['log'] = str(LineNumber)
                else:
                    if LineNumber:
                        log['line'] = str(LineNumber)
                    log['log'] = message
                    
            else:
                log['error_type'] = message
                log['error_line'] = LineNumber
                log['error_log'] = ErrorDescription

        except Exception as o_Error:
            log['time'] = time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', time.localtime()
            )
            log['level'] = 'ERROR'
            log['error_type'] = str(type(o_Error).__name__)
            log['error_line'] = 'monitoring.' + str(
                sys.exc_info()[-1].tb_lineno
            )
            log['error_log'] = o_Error

        print(json.dumps(log))

        time.sleep(1)

    def print_critical_logs(self):
        print(self.critical_logs)


log=Logs()


