import pandas as pd
import preprocessing
import BASE_LOG_ANALYSE
from multiprocess import Pool
from functools import partial
import datetime

data = dict()
entropy = dict()
vectorizer = None


def load_data_log_entity(folder, entity_name, hostname):
    data1 = {'timestamp': [], 'message': []}
    for filename in BASE_LOG_ANALYSE.get_file_list_by_filename_filter(folder, entity_name):
        # Get file content - Start
        if '.gz' in filename:
            content = BASE_LOG_ANALYSE.read_gzip_file(filename)
        elif 'pfe' in filename:
            pass
        else:
            with open(filename, 'r', encoding="latin-1") as f_in:
                content = f_in.read()
        # Get file content - End
        # Parsing data to get time and info - Start
        if 'messages' in filename:
            for line in content.split('\n'):
                tokens = line.split(hostname)
                if "logfile turned over due" in line:
                    pass
                else:
                    if len(tokens) == 2:
                        if '<' in tokens[0]:
                            tokens[0] = tokens[0].split()[1]
                        data1['timestamp'].append(tokens[0])
                        data1['message'].append(tokens[1])
        elif 'pfe' in filename:
            pass
        else:
            for line in content.split('\n'):
                tokens = line.split(hostname)
                if "logfile turned over due" in line:
                    pass
                else:
                    if len(tokens) == 2:
                        if '<' in tokens[0]:
                            tokens[0] = tokens[0].split()[1]
                        data1['timestamp'].append(tokens[0])
                        data1['message'].append(tokens[1])
        # Parsing data to get time and info - End
    return pd.DataFrame(data1)

def training_data(folder, start, end, hostname):
    global entropy, vectorizer, data
    s = datetime.datetime.now()
    df = load_data_log_entity(folder, 'messages*', hostname)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df.sort_values(by=['timestamp'], inplace=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    data['message'] = df
    t = datetime.datetime.now()
    print("Load data in {} second".format(t-s))
    training_data = {k: v[(v['timestamp'] >= start) & (v['timestamp'] < end)].copy(deep=True) for k, v in data.items()}
    entropy, vectorizer = preprocessing.preprocess_training_data(training_data)
    print("Train data in {} second".format(datetime.datetime.now() - t))
    return "Training done!"

def speed_up(time_filter, data, vectorizer, entropy):
    data_filter = { k: v[(v['timestamp'] >= str(time_filter[0])) & (v['timestamp'] < str(time_filter[1]))] for k, v in data.items() }
    return {'timestamp': time_filter[0].to_timestamp(), 'score': preprocessing.calculate_score(data_filter, vectorizer, entropy)['message']}

def speed_up_split(filtered_data, vectorizer, entropy):
    if len(filtered_data[1] == 0):
        return {'timestamp': filtered_data[0].to_datetime64(), 'score': 0}
    else:
        return {'timestamp': filtered_data[0].to_datetime64(), 'score': preprocessing.calculate_score({'message': filtered_data[1]}, vectorizer, entropy)['message']}

def split_chunks_and_calculate_score(num_core, data, start, end, vectorizer, entropy, chunk_size='10s'):
    # period = pd.period_range(start=start, end=end, freq=chunk_size)
    # input_args = []
    # for i in range(len(period) - 1):
    #     input_args.append([period[i], period[i + 1]])
    
    testing = data['message'][(data['message']['timestamp'] >= start) & (data['message']['timestamp'] < end)]
    input_args = [[n, g] for n, g in testing.groupby(pd.Grouper(key='timestamp',freq='10s'))]
    pool = Pool(int(num_core))
    chunks = pool.map(partial(speed_up_split, vectorizer=vectorizer, entropy=entropy), input_args)
    pool.close()
    pool.join()
    return chunks

def testing_data(start, end):
    global entropy, vectorizer, data
    score_chunks = split_chunks_and_calculate_score(3, data, start, end, vectorizer, entropy)
    score = pd.DataFrame(score_chunks)
    testing = {k: v[(v['timestamp'] >= start) & (v['timestamp'] < end)] for k, v in data.items()}['message']
    grouped = testing.groupby(pd.Grouper(key='timestamp', axis=0, freq='H')).count()
    grouped = grouped.reset_index()
    grouped.columns = ['timestamp', 'count']
    # testing['timestamp'] = testing['timestamp'].dt.tz_convert(None)
    return score, grouped, testing


if __name__ == "__main__":
    training_data('/home/kien/mycode/varlog_NTH9205CRT10_02062023/var/log', '2023-05-01', '2023-05-30', 'NTH9205.PECD.MX2020.02_RE0')
    # x, y, z = testing_data('2023-06-01', '2023-06-03')