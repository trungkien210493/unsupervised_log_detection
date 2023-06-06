import gradio as gr
import datetime
import os
import preprocessing
import BASE_LOG_ANALYSE
import pandas as pd
# from multiprocess import Pool
from ray.util.multiprocessing import Pool
import urllib3
from functools import partial
urllib3.disable_warnings()
import plotly.express as px


data = dict()
entropy = dict()
vectorizer = None

def training_data(folder, start, end, hostname):
    df = load_data_log_entity(folder, 'messages*', hostname)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['timestamp'], inplace=True)
    data['message'] = df
    training_data = {k: v[(v['timestamp'] >= start) & (v['timestamp'] < end)] for k, v in data.items()}
    global entropy, vectorizer
    entropy, vectorizer = preprocessing.preprocess_training_data(training_data)
    return "Training done!"

def inspect_data(start, end):
    inspect = {k: v[(v['timestamp'] >= start) & (v['timestamp'] < end)] for k, v in data.items()}['message']
    return inspect

def testing_data(start, end, num_core, threshold):
    score_chunks = split_chunks_and_calculate_score(num_core, data, start, end, vectorizer, entropy)
    score = pd.DataFrame(score_chunks)
    p = px.line(score, x='timestamp', y='score')
    anomaly = score[score['score'] >= threshold]
    testing = {k: v[(v['timestamp'] >= start) & (v['timestamp'] < end)] for k, v in data.items()}['message']
    grouped = testing.groupby(pd.Grouper(key='timestamp', axis=0, freq='H')).count()
    p2 = px.line(grouped)
    return p, anomaly, p2

def load_data_log_entity(folder, entity_name, hostname):
    data1 = {'timestamp': [], 'message': []}
    for filename in BASE_LOG_ANALYSE.get_file_list_by_filename_filter(folder, entity_name):
        # Get file content - Start
        if '.gz' in filename:
            content = BASE_LOG_ANALYSE.read_gzip_file(filename)
        else:
            with open(filename, 'r', encoding="latin-1") as f_in:
                content = f_in.read()
        # Get file content - End
        # Parsing data to get time and info - Start
        if 'messages' in filename:
            for line in content.split('\n'):
                tokens = line.split(hostname)
                if len(tokens) == 2:
                    if '<' in tokens[0]:
                        tokens[0] = tokens[0].split()[1]
                    data1['timestamp'].append(tokens[0])
                    data1['message'].append(tokens[1])
        else:
            for line in content.split('\n'):
                tokens = line.split(hostname)
                if len(tokens) == 2:
                    if '<' in tokens[0]:
                        tokens[0] = tokens[0].split()[1]
                    data1['timestamp'].append(tokens[0])
                    data1['message'].append(tokens[1])
        # Parsing data to get time and info - End    
    return pd.DataFrame(data1)


def speed_up(time_filter, data, vectorizer, entropy):
    data_filter = { k: v[(v['timestamp'] >= str(time_filter[0])) & (v['timestamp'] < str(time_filter[1]))] for k, v in data.items() }
    return {'timestamp': time_filter[0].to_timestamp(), 'score': preprocessing.calculate_score(data_filter, vectorizer, entropy)['message']}

def split_chunks_and_calculate_score(num_core, data, start, end, vectorizer, entropy, chunk_size='10s'):
    period = pd.period_range(start=start, end=end, freq=chunk_size)
    input_args = []
    for i in range(len(period) - 1):
        input_args.append([period[i], period[i + 1]])
    pool = Pool(processes=int(num_core))
    chunks = pool.map(partial(speed_up, data=data, vectorizer=vectorizer, entropy=entropy), input_args)
    return chunks

demo = gr.Blocks()
with demo:
    gr.Markdown(
        r"<h1>Please choose the training period and train model by this data to reuse in test data. Please change the datetime format if the current datetime format does not match with '_message' log file</h1>"
    )
    with gr.Row():
        with gr.Column(scale=2):
            dir_path = gr.Textbox(
                value=os.path.abspath(os.getcwd()),
                interactive=True,
                label="Log directory"
            )
        with gr.Column(scale=1):
            hostname = gr.Textbox(
                value="",
                interactive=True,
                label="Hostname in log file, please check the _message file"
            )
    with gr.Row():
        start_training = gr.Textbox(
            label="Start training time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
        end_training = gr.Textbox(
            label="End training time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
    output = gr.Textbox(label="Training result")
    btn1 = gr.Button(value="Train data")
    btn1.click(fn=training_data, inputs=[dir_path, start_training, end_training, hostname], outputs=output)
    with gr.Row():
        start_testing = gr.Textbox(
            label="Start testing time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
        end_testing = gr.Textbox(
            label="End testing time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
    with gr.Row():
        num_core = gr.Number(
            label="Number CPU core (use more cores to speed up)",
            interactive=True,
            value=3
        )
        threshold = gr.Number(
            label="Threshold to consider error",
            interactive=True,
            value=5
        )
    btn2 = gr.Button(value="Check testing data")
    pl = gr.Plot()
    gr.Markdown("<h1>Possible error</h1>")
    ano = gr.DataFrame()
    gr.Markdown("<h1>Total log count per hour</h1>")
    pl2 = gr.Plot()
    btn2.click(fn=testing_data, inputs=[start_testing, end_testing, num_core, threshold], outputs=[pl, ano, pl2])
    gr.Markdown("<h1>Inspect data</h1>")
    with gr.Row():
        start_inspect = gr.Textbox(
            label="Start inspect time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
        end_inspect = gr.Textbox(
            label="End inspect time",
            interactive=True,
            value=str(datetime.datetime.now())
        )
    btn3 = gr.Button(value="Inspect data in above interval")
    raw = gr.DataFrame()
    btn3.click(fn=inspect_data, inputs=[start_inspect, end_inspect], outputs=raw)
    

if __name__ == "__main__":
    demo.launch()