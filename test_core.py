import os
import time
import BASE_LOG_ANALYSE
import pandas as pd
import gzip
import zlib
import re
import mysql.connector
from multiprocesspandas import applyparallel


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
if __name__ == '__main__':
    conn = mysql.connector.connect(host='10.98.100.107', user='juniper', password='juniper@123', database='svtech_log')
    cursor = conn.cursor()
    cursor.execute("SELECT regex_pattern, template_id FROM template")
    result = cursor.fetchall()

    s1 = time.time()
    with open('/home/kien/NAN0322AGG02_varlog_20180904/var/log/messages', 'r', encoding="latin-1") as f_in:
        content = f_in.read()
    s2 = time.time()
    parse = []
    x = [r".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')) for pattern in result]
    y = [pattern[1] for pattern in result]
    # print(x)
    # for pattern in result:
    #     res = re.findall(r".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')), content)
    #     if len(res) > 0:
    #         print("Found {}".format(pattern[1]))
    # pool = Pool(4)
    # chunks = pool.map(partial(re.findall, string=content), x)
    # pool.close()
    # pool.join()
    for line in content.split('\n'):
        [timestamp, log_info, pri_code] = BASE_LOG_ANALYSE.parsing_line(line)
        parse.append([timestamp, log_info, pri_code])
    df = pd.DataFrame(parse, columns=['timestamp', 'message', 'pri'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    df.drop(['pri'], axis=1, inplace=True)
    def replace_x(df, x, y):
        s = dict(zip(x, y))
        for k, v in s.items():
            if re.match(k, df['message']):
                df['template_id'] = v
                break
        return df
        
    df['template_id'] = 'Unknown'
    df = df.apply_parallel(replace_x, x=x, y=y, num_processes=4)
    print(df.head())
    s3 = time.time()
    print("Read file in: {} seconds".format(s2 - s1))
    print("Regex file in: {} seconds".format((s3 - s2)))
    df.to_csv("testing.csv")