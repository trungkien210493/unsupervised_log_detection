import panel as pn
import pandas as pd
import os
import datetime as dt
import altair as alt
import preprocessing
import BASE_LOG_ANALYSE
import nltk
from multiprocess import Pool
from functools import partial
pn.extension('vega')
pn.extension('tabulator')
pn.extension(notifications=True)
from pyunpack import Archive
import polars as pl
alt.data_transformers.disable_max_rows()
nltk.download('wordnet')
import mysql.connector
from datasource import es_connection, ticket_db, num_core, pattern_dict, data_path, MAX_SIZE_MB
import re
import urllib3
import syslog_rust
urllib3.disable_warnings()
import multiprocessing
from rich.progress import track

# Global variable
verify_data = None
entropy = dict()
vectorizer = None
regex_dict = dict()
template_id_df = None
# UI components - Start
# Sidebar
file_input = pn.widgets.FileInput(accept='.tar,.tar.gz,.zip,.rar,.tgz')
progress = pn.indicators.Progress(active=False)
loading = pn.indicators.LoadingSpinner(width=20, height=20, value=False, visible=False)
sidebar = pn.layout.WidgetBox(
    file_input,
    progress
)

# Main
# Analysis tab - Start
brush = alt.selection_interval(
    name="brush",
    encodings=["x"],
    on="[mousedown[event.altKey], mouseup] > mousemove",
    translate="[mousedown[event.altKey], mouseup] > mousemove!",
    zoom="wheel![event.altKey]",
)

interaction = alt.selection_interval(
    bind="scales",
    on="[mousedown[event.shiftKey], mouseup] > mousemove",
    translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
    zoom="wheel![event.shiftKey]",
)

empty_score = pd.DataFrame({'timestamp': [], 'score': []})
score_chart = alt.Chart(empty_score).mark_line().encode(
    x='timestamp',
    y='score',
    tooltip=['timestamp', 'score']
).properties(
    width=600,
    height=320
).add_params(
    brush,
    interaction
)
score_panel = pn.pane.Vega(score_chart, margin=5)

empty_log_count = pd.DataFrame({'timestamp': [], 'count': []})
log_count_chart = alt.Chart(empty_log_count).mark_line().encode(
    x='timestamp',
    y='count',
    tooltip=['timestamp', 'count']
).properties(
    width=600,
    height=320
).interactive()

log_count_panel = pn.pane.Vega(log_count_chart, margin=5)
error = pn.widgets.Tabulator(sizing_mode="stretch_both", margin=5, page_size=5, pagination='remote')
show_log = pn.widgets.Tabulator(styles={"font-size": "10pt"}, sizing_mode="stretch_both", margin=5, pagination=None)
training_period = pn.widgets.DatetimeRangePicker(name="Training period", align='end')
train_but = pn.widgets.Button(name='Train', sizing_mode='stretch_width', align='end')
testing_period = pn.widgets.DatetimeRangePicker(name="Testing period", align='end')
threshold = pn.widgets.IntSlider(name='Threshold', start=4, end=10, step=1, value=5, align='end')
test_but = pn.widgets.Button(name='Test', sizing_mode='scale_width', align='end')
# Analysis tab - End
# Check KB tab - Start
filter_file = pn.widgets.CheckBoxGroup(
    name='Log files', options=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    value=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    inline=True,
    align='end'
)
check_kb_but = pn.widgets.Button(name="Check", align='end')
check_kb_time = pn.widgets.DatetimeRangePicker(name="Time filter", align='end')
show_kb = pn.widgets.Tabulator(styles={"font-size": "10pt"}, sizing_mode="stretch_both", margin=5, pagination=None)
kb_tab = pn.Column(
    pn.Row(filter_file, check_kb_time, check_kb_but),
    show_kb
)
# Check KB tab - End
# Log pattern tab - Start
filter_file_pattern = pn.widgets.CheckBoxGroup(
    name='Log files', options=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    value=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    inline=True,
    align='end'
)
get_data_but = pn.widgets.Button(name="Get log pattern", align='end')
count_template_id = alt.Chart(pd.DataFrame({'timestamp': [], 'count': [], 'template_id': []})).mark_line().encode(
    x='timestamp',
    y='count',
    color='template_id:N',
    tooltip=['timestamp', 'count', 'template_id']
).properties(
    width=700,
    height=720
).add_params(
    brush,
    interaction
)

count_panel = pn.pane.Vega(count_template_id, margin=5, min_width=700)
show_log_pattern = pn.widgets.Tabulator(styles={"font-size": "9pt"}, 
                                        layout='fit_data_table', 
                                        sizing_mode="stretch_both", 
                                        hidden_columns=['index'],
                                        min_width=800, pagination=None,)
filter_time_log_pattern = pn.widgets.DatetimeRangePicker(name="Time filter", align='end')
log_pattern_tab = pn.Column(
    pn.Row(filter_file_pattern, filter_time_log_pattern, get_data_but),
    pn.Row(count_panel, show_log_pattern)
)
# Log pattern tab - End
# Ticket tab - Start
tag_name = pn.widgets.TextInput(name="Tag name")
ticket_time = pn.widgets.DatetimeRangePicker(name="Error time")
customer = pn.widgets.Select(name="Customer", options=['viettel', 'metfone', 'unitel', 'movitel', 'nnpt', 'mobifone', 'cmc', 'natcom', 'ftel'])
tag_optional = pn.widgets.TextInput(name="Optional tag")
description = pn.widgets.TextAreaInput(name="Description", height=200)
save_but = pn.widgets.Button(name="Save")
ticket_tab = pn.Column(
    tag_name,
    ticket_time,
    customer,
    tag_optional,
    description,
    save_but,
    width=1500
)
# Ticket tab - End

main = pn.Tabs(
        ('Analysis', pn.Column(
            pn.Row(training_period, train_but, testing_period, test_but, threshold),
            pn.GridBox(ncols=2, nrows=2,
                        objects=[
                            pn.Column(pn.pane.Markdown("# Score panel"), score_panel),
                            pn.Column(pn.pane.Markdown("# Possible error"), error, sizing_mode="stretch_both"),
                            pn.Column(pn.pane.Markdown("# Total document count per hour"), log_count_panel),
                            pn.Column(pn.pane.Markdown("# Raw log"), show_log, sizing_mode="stretch_both")], 
                        sizing_mode="stretch_both"
        ))),
        ('Check KB', kb_tab),
        ('Log pattern', log_pattern_tab),
        ('Save ticket', ticket_tab),
)
# Main page
main_page = pn.template.MaterialTemplate(
    title='Syslog Analysis',
    sidebar=sidebar,
    busy_indicator=loading,
    main=main
)

# UI components - End
# Component logic - Start
def reset(event):
    global data_path
    file_input.disabled = False
    progress.active = True
    progress.active = False
    pn.state.notifications.info('Upload file done, try to extract file.', duration=2000)
    if not os.path.exists(data_path):
        try:
            os.makedirs(data_path)
        except Exception as e:
            pn.state.notifications.error('Data path is empty, but can not create', duration=2000)
    # Create directory to save file upload and extract result
    upload_path = os.path.join(data_path, 'file_upload')
    if not os.path.exists(upload_path):
        try:
            os.makedirs(upload_path)
        except Exception as e:
            pn.state.notifications.error("Can't create the directory to store file upload", duration=2000)
    # Create directory to save extract data
    extract_path = os.path.join(data_path, 'extracted')
    if not os.path.exists(extract_path):
        try:
            os.makedirs(extract_path)
        except Exception as e:
            pn.state.notifications.error("Can't create the directory to store extracted data", duration=2000)
    # Save file upload
    path = os.path.join(upload_path, file_input.filename)
    file_input.save(path)
    # Extract file upload
    file_name, file_extension = os.path.splitext(file_input.filename)
    if not os.path.exists(os.path.join(extract_path, file_name)):
        try:
            os.makedirs(os.path.join(extract_path, file_name))
        except Exception as e:
            pn.state.notifications.error("Error to create directory inside extracted directory", duration=2000)
    try:
        Archive(path).extractall(os.path.join(extract_path, file_name))
        pn.state.notifications.info('Extract file done.', duration=2000)
    except:
        pn.state.notifications.error("Extract error! Check your upload file or contact admin", duration=2000)
    
file_input.jscallback(
    args={"progress": progress},
    value="""
        progress.active = true;
        source.disabled = true;
    """
)
file_input.param.watch(reset, 'value')

def load_display(x):
    if(x=='on'):
        loading.value=True
        loading.visible=True
    if(x=='off'):
        loading.value=False
        loading.visible=False

def get_saved_data_path():
    global data_path
    extract_path = os.path.join(data_path, 'extracted')
    file_name, file_extension = os.path.splitext(file_input.filename)
    return os.path.join(extract_path, file_name)
    
# Analysis tab - Start
def training_data(folder, start, end):
    global entropy, vectorizer
    list_file = []
    for filename in BASE_LOG_ANALYSE.get_file_list_by_filename_filter(folder, 'messages*'):
        list_file.append(filename)
    try:
        train_data = syslog_rust.processing_log(list_file, start, end)
        if len(train_data) == 0:
            pn.state.notifications.warning("There is no data in current time filter")
        else:
            df = pd.DataFrame(train_data)[['time', 'log']].rename(columns={"time": "timestamp", "log": "message"})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(by=['timestamp'], inplace=True)
            entropy, vectorizer = preprocessing.preprocess_training_data({'message': df})
            pn.state.notifications.info("Training done!")
    except Exception as e:
        pn.state.notifications.error("Training error due to: {}".format(e))

def speed_up_split(filtered_data, vectorizer, entropy):
    if len(filtered_data[1]) == 0:
        return {'timestamp': filtered_data[0].to_pydatetime(), 'score': 0}
    else:
        return {'timestamp': filtered_data[0].to_pydatetime(), 'score': preprocessing.calculate_score({'message': filtered_data[1]}, vectorizer, entropy)['message']}

def testing_data(folder, start, end):
    global entropy, vectorizer, num_core, verify_data
    list_file = []
    for filename in BASE_LOG_ANALYSE.get_file_list_by_filename_filter(folder, 'messages*'):
        list_file.append(filename)
    try:
        pn.state.notifications.info("Verify data in testing period")
        test_data = syslog_rust.processing_log(list_file, start, end)
        if len(test_data) == 0:
            pn.state.notifications.warning("There is no data in current time filter")
        else:
            df = pd.DataFrame(test_data)[['time', 'log']].rename(columns={"time": "timestamp", "log": "message"})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(by=['timestamp'], inplace=True)
            input_args = [[n, g] for n, g in df.groupby(pd.Grouper(key='timestamp',freq='10s'))]
            pool = Pool(int(num_core))
            chunks = pool.map(partial(speed_up_split, vectorizer=vectorizer, entropy=entropy), input_args)
            pool.close()
            pool.join()
            score = pd.DataFrame(chunks)
            grouped = df.groupby(pd.Grouper(key='timestamp', axis=0, freq='H')).count()
            grouped = grouped.reset_index()
            grouped.columns = ['timestamp', 'count']
            score_chart.data = score
            score_panel.param.trigger('object')
            log_count_chart.data = grouped
            log_count_panel.param.trigger('object')
            # To convert from utc time to local time display (display only)
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            show_log.value = df
            verify_data = df
            error.value = score[score['score'] > threshold.value]
    except Exception as e:
        pn.state.notifications.error("Testing error due to: {}".format(e))

def train_but_click(event):
    load_display('on')
    pn.state.notifications.info("Start training data")
    training_data(get_saved_data_path(), str(training_period.value[0]), str(training_period.value[1]))
    load_display('off')

def test_but_click(event):
    load_display('on')
    testing_data(get_saved_data_path(), str(testing_period.value[0]), str(testing_period.value[1]))
    load_display('off')

train_but.on_click(train_but_click)
test_but.on_click(test_but_click)

def callback_error(target, event):
   target.value = score_chart.data[score_chart.data['score'] > event.new]

threshold.link(error, callbacks={'value': callback_error})

def inspect_data(start, end):
    global verify_data
    inspect = verify_data[(verify_data['timestamp'] >= start) & (verify_data['timestamp'] < end)].copy()
    return inspect

def filtered_score(selection):
    if not selection:
        pass
    else:
        s = dt.datetime.fromtimestamp(selection['timestamp'][0]/1000)
        e = dt.datetime.fromtimestamp(selection['timestamp'][1]/1000)
        show_log.value = inspect_data(str(s), str(e))
        filter_score = score_chart.data[(score_chart.data['timestamp'] >= str(s)) & (score_chart.data['timestamp'] < str(e))].copy()
        filter_score['timestamp'] = filter_score['timestamp'].dt.tz_localize(None)
        error.value = filter_score[filter_score['score'] > threshold.value]

pn.bind(filtered_score, score_panel.selection.param.brush, watch=True)
# Analysis tab - End
# Check KB tab - Start

def find_kb(row):
    global pattern_dict
    for kb_id, regex_pattern in pattern_dict.items():
        if re.match('.*{}.*'.format(regex_pattern), row):
            return kb_id
    return None

def parallel_apply(function, column):
    global num_core
    with multiprocessing.get_context("fork").Pool(num_core) as pool:
        return pl.Series(pool.imap(function, track(column)))

def check_kb_click(event):
    load_display('on')
    process_files = []
    for filterd in filter_file.value:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    process_data = syslog_rust.processing_log(process_files, str(check_kb_time.value[0]), str(check_kb_time.value[1]))
    if len(process_data) == 0:
        pn.state.notifications.warning("There is no data in current time filter")
    else:
        try:
            df = pl.DataFrame(process_data).lazy()
            df = df.with_columns(kb=pl.col("log").map_batches(lambda col: parallel_apply(find_kb, col))).collect()
            df = df.filter(pl.col('kb').is_not_null())
            if len(df) > 0:
                show_kb.value = df.to_pandas()[['kb', 'time', 'filename', 'log']]
            else:
                show_kb.value = pd.DataFrame([{'result': 'there is no log match KB'}])
        except Exception as e:
            pn.state.notifications.error("Check kb error due to: {}".format(e))
    load_display('off')
check_kb_but.on_click(check_kb_click)
# Check KB tab - End
# Log pattern tab - Start
def find_template_id(row):
    global regex_dict
    for regex_pattern, template_id in regex_dict.items():
        if re.match(regex_pattern, row):
            return template_id
    return None

def get_data_click(event):
    load_display('on')
    # Get template from mysql
    conn = mysql.connector.connect(
        host=ticket_db["host"],
        port=ticket_db["port"],
        user=ticket_db["user"],
        password=ticket_db["password"],
        database="svtech_log"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT regex_pattern, template_id FROM template")
    result = cursor.fetchall()
    global regex_dict, template_id_df
    regex_dict = {".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')): pattern[1] for pattern in result }
    cursor.close()
    conn.close()
    process_files = []
    for filterd in filter_file.value:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    process_data = syslog_rust.processing_log(process_files, str(filter_time_log_pattern.value[0]), str(filter_time_log_pattern.value[1]))
    if len(process_data) == 0:
        pn.state.notifications.warning("There is no data in current time filter")
    else:
        try:
            df = pl.DataFrame(process_data).lazy()
            df = df.with_columns(template_id=pl.col("log").map_batches(lambda col: parallel_apply(find_template_id, col))).collect()
            template_id_df = df.to_pandas()
            template_id_df.fillna({'template_id': 'Unknown'}, inplace=True)
            template_id_df['time'] = pd.to_datetime(template_id_df['time'])
            template_id_df['time'] = template_id_df['time'].dt.tz_localize(None)
            template_id_df.rename(columns={"time": "timestamp"}, inplace=True)
            # Add log
            show_log_pattern.value = template_id_df[['timestamp', 'template_id', 'filename', 'log']]
            # Add panel
            template_id_df['date_hour'] = template_id_df['timestamp'].dt.to_period('H')
            top10_template_id = template_id_df['template_id'].value_counts().nlargest(10).index
            df_top10 = template_id_df[template_id_df['template_id'].isin(top10_template_id)]
            top10_per_date_hour = df_top10.groupby(['date_hour', 'template_id']).size().reset_index(name='count')
            top10_per_date_hour['date_hour'] = top10_per_date_hour['date_hour'].dt.to_timestamp()
            top10_per_date_hour.columns = ['timestamp', 'template_id', 'count']
            top10_per_date_hour['timestamp'] = top10_per_date_hour['timestamp'].dt.tz_localize('Asia/Ho_Chi_Minh')
            count_template_id.data = top10_per_date_hour
            count_panel.param.trigger('object')
        except Exception as e:
            pn.state.notifications.error("Tag template id error due to: {}".format(e))
    load_display('off')

get_data_but.on_click(get_data_click)

def filtered_log(selection):
    global template_id_df
    if not selection:
        pass
    else:
        s = dt.datetime.fromtimestamp(selection['timestamp'][0]/1000)
        e = dt.datetime.fromtimestamp(selection['timestamp'][1]/1000)
        show_log_pattern.value = template_id_df[(template_id_df['timestamp'] >= str(s)) & (template_id_df['timestamp'] < str(e))][['timestamp', 'template_id', 'filename', 'log']]
pn.bind(filtered_log, count_panel.selection.param.brush, watch=True)
# Log pattern tab - End
# Save ticket - Start
def save_but_click(event):
    try:
        cnx = mysql.connector.connect(
            host=ticket_db["host"],
            port=ticket_db["port"],
            user=ticket_db["user"],
            password=ticket_db["password"],
            database="svtech_log"
        )
        cursor = cnx.cursor()
        table_name = 'ticket'
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()
        if not result:
            table_creation_query = f'''
            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `file_name` text DEFAULT NULL,
                `tag_name` text DEFAULT NULL,
                `start_time` datetime DEFAULT NULL,
                `stop_time` datetime DEFAULT NULL,
                `customer` text DEFAULT NULL,
                `tag_optional` text DEFAULT NULL,
                `description` longtext DEFAULT NULL
            )
            '''
            cursor.execute(table_creation_query)
        insert_query = '''
        INSERT INTO ticket (file_name, tag_name, start_time, stop_time, customer, tag_optional, description)
        VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');
        '''.format(file_input.filename, tag_name.value, str(ticket_time.value[0]), str(ticket_time.value[1]), customer.value, tag_optional.value, description.value)
        cursor.execute(insert_query)
        cnx.commit()
        cursor.close()
        cnx.close()
        pn.state.notifications.info("Save succesfully!")
    except Exception as e:
        pn.state.notifications.error("Can NOT save ticket due to: {}".format(e))

save_but.on_click(save_but_click)
# Save ticket - End
# Component logic - End

ROUTES = {
    "app_panel": main_page,
}
pn.serve(ROUTES,
         port=3001,
         address='0.0.0.0',
         websocket_origin='*',
         websocket_max_message_size=MAX_SIZE_MB*1024*1024,
         http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1024}
        )