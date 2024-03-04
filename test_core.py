import os
import time
import BASE_LOG_ANALYSE
import pandas as pd
import gzip
import zlib
import re
import mysql.connector
from multiprocesspandas import applyparallel
from concurrent.futures import ThreadPoolExecutor
import syslog_rust
from pygrok import Grok
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.palettes import Category20


def is_valid_file(filepath, start_time, stop_time):
    modification_time = os.path.getmtime(filepath)
    if modification_time >= start_time:
        content = ''
        if '.gz' in filepath:
            content = BASE_LOG_ANALYSE.read_gzip_file(filepath)
        elif 'pfe' in filepath:
            pass
        else:
            with open(filepath, 'r', encoding="latin-1") as f_in:
                content = f_in.read()
        for line in content.split('\n'):
            [timestamp, log_info, pri_code] = BASE_LOG_ANALYSE.parsing_line(line)
            t = pd.to_datetime(timestamp, errors='coerce', utc=True)
            if not pd.isnull(t):
                break
        if pd.isnull(t):
            return False
        else:
            if t.timestamp() > stop_time:
                return False
            else:
                return True
    else:
        return False

def parse_single_line_rfc5424(line):
    if not line['log'].startswith("<"):
        pass
    else:
        try:
            res = syslog_rust.parse_syslog_rfc5424(line['log'].strip())
            for k, v in res.items():
                line[k] = v
        except:
            pass

custom_pattern = {
    'DATESTAMP_FULL': "%{MONTH} %{MONTHDAY} %{TIME} %{YEAR}",
    'DATESTAMP_FULL2': "%{MONTH}  %{MONTHDAY} %{TIME} %{YEAR}",
    'DATESTAMP_NOTYEAR': "%{MONTH} %{MONTHDAY} %{TIME}",
    'JUNHOSTNAME': "[a-zA-Z0-9\_\-\[\.]{1,}",
    'PROCESSNAME': "[a-zA-Z0-9\/\:\_\-\.]{1,}",
    'PROCESSID': "(?<=\[)[0-9_]{1,}",
    'FACILITYNAME': "[a-zA-Z]{1,}",
    'SEVERITYCODE': "[0-9_]{1,}",
    'EVENTNAME': "[a-zA-Z0-9\_]{1,}",
    'MESSAGE': "(?<=\:\s).*"
}
pattern = """(%{DATESTAMP_FULL:junos_time}|%{DATESTAMP_FULL2:junos_time}|%{DATESTAMP_NOTYEAR:junos_time})  %{JUNHOSTNAME:junos_hostname} ((%{PROCESSNAME:junos_procsname}\[%{PROCESSID:junos_procsid}\]\:)|(%{PROCESSNAME:junos_procsname}\:)|(\:)) ((\%%{FACILITYNAME:junos_facilityname}\-%{SEVERITYCODE:junos_severitycode}\-%{EVENTNAME:junos_eventname}\:)|(\%%{FACILITYNAME:junos_facilityname}\-%{SEVERITYCODE:junos_severitycode}\:)|(\%%{FACILITYNAME:junos_facilityname}\-%{SEVERITYCODE:junos_severitycode}\-\:)) %{MESSAGE:junos_msg}"""
grok = Grok(pattern, custom_patterns=custom_pattern)
reversed_severity = {
    '0': 'emergency',
    '1': 'alert',
    '2': 'critical',
    '3': 'error',
    '4': 'warning',
    '5': 'notice',
    '6': 'informational',
    '7': 'debug'
}

def parse_pygrok(line):
    global grok
    res = grok.match(line['log'])
    if res is not None:
        line['junos_facilityname'] = res['junos_facilityname']
        line['junos_severitycode'] = reversed_severity[res['junos_severitycode']]
        line['junos_eventname'] = res['junos_eventname']

if __name__ == '__main__':
    # conn = mysql.connector.connect(host='10.98.100.107', user='juniper', password='juniper@123', database='svtech_log')
    # cursor = conn.cursor()
    # cursor.execute("SELECT regex_pattern, template_id FROM template")
    # result = cursor.fetchall()

    # s1 = time.time()
    # with open('/home/kien/NAN0322AGG02_varlog_20180904/var/log/messages', 'r', encoding="latin-1") as f_in:
    #     content = f_in.read()
    # s2 = time.time()
    # parse = []
    # x = [r".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')) for pattern in result]
    # y = [pattern[1] for pattern in result]
    # print(x)
    # for pattern in result:
    #     res = re.findall(r".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')), content)
    #     if len(res) > 0:
    #         print("Found {}".format(pattern[1]))
    # pool = Pool(4)
    # chunks = pool.map(partial(re.findall, string=content), x)
    # pool.close()
    # pool.join()
    # for line in content.split('\n'):
    #     [timestamp, log_info, pri_code] = BASE_LOG_ANALYSE.parsing_line(line)
    #     parse.append([timestamp, log_info, pri_code])
    # df = pd.DataFrame(parse, columns=['timestamp', 'message', 'pri'])
    # df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    # df = df.dropna(subset=['timestamp'])
    # df.drop(['pri'], axis=1, inplace=True)
    # def replace_x(df, x, y):
    #     s = dict(zip(x, y))
    #     for k, v in s.items():
    #         if re.match(k, df['message']):
    #             df['template_id'] = v
    #             break
    #     return df
        
    # df['template_id'] = 'Unknown'
    # df = df.apply_parallel(replace_x, x=x, y=y, num_processes=4)
    # print(df.head())
    # s3 = time.time()
    # print("Read file in: {} seconds".format(s2 - s1))
    # print("Regex file in: {} seconds".format((s3 - s2)))
    # df.to_csv("testing.csv")
    # process_data = syslog_rust.processing_log(["/home/kien/messages"], "2021-11-28", "2021-11-30")
    # print(process_data[0:5])
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(parse_single_line_rfc5424, process_data)
    # print(process_data[0:5])
    process_data = syslog_rust.processing_log(["/var/tmp/for_unsupervised_report/extracted/NAN0322AGG02_varlog_20180904/var/log/messages"], "2018-08-25", "2021-08-30")
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(parse_pygrok, process_data)
    df = pd.DataFrame(process_data)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'], inplace=True)
    df.set_index('time', inplace=True)
    count = df.resample('1min')['junos_facilityname'].value_counts().unstack(fill_value=0)
    count = count.asfreq('1min', fill_value=0)
    source = ColumnDataSource(count)

    # Create a Bokeh figure
    p = figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=700, tools='pan,box_zoom,reset')

    # Define a color palette with enough colors for all columns
    colors = Category20[len(count.columns)]

    # Create a Legend
    legend = Legend(items=[(column, [p.line(x='time', y=column, source=source, line_width=2, line_color=colors[i])]) for i, column in enumerate(count.columns)])

    # Add the Legend to the plot
    p.add_layout(legend, 'right')
    p.legend.click_policy = 'hide'


    # Customize the plot
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Count'

    # Show the plot
    show(p)