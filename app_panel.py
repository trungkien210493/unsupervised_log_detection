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
pn.extension('texteditor')
import polars as pl
alt.data_transformers.disable_max_rows()
try:
    nltk.find('corpora/wordnet')
except:
    nltk.download('wordnet')
import mysql.connector
from datasource import ticket_db, num_core, pattern_dict, data_path, surreal_db, es_connection
import re
import urllib3
import urllib
import syslog_rust
urllib3.disable_warnings()
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pygrok import Grok
import gensim
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, DatetimeTickFormatter
from bokeh.palettes import Category20
import subprocess
from datetime import datetime, timedelta
from surrealdb import SurrealHTTP, Surreal
from elasticsearch import Elasticsearch
import tag_template_rust
import numpy as np
import asyncio
import hashlib

pn.extension(nthreads=4)
# Global variable
verify_data = None
entropy = dict()
vectorizer = None
regex_dict = dict()
parse_data = None
dictionary = gensim.corpora.Dictionary.load(
    "model_26_Dec_2024/dictionary/template_dictionary.dict"
)
lda_model = gensim.models.LdaMulticore.load(
    "model_26_Dec_2024/model/template_lda.model"
)
# UI components - Start
# Sidebar
def find_file(path):
    if os.path.exists(path):
        return os.listdir(path)
    else:
        return []
    
upload_or_select = pn.widgets.RadioBoxGroup(name="upload or select", options=['upload', 'select'], inline=True, value='upload')
extract_direct = pn.widgets.MultiChoice(name="Host", options=find_file(os.path.join(data_path, 'extracted')), max_items=1)
case_id = pn.widgets.TextInput(name='Case', placeholder='Enter your case id ...')
file_input = pn.widgets.FileInput(accept='.tar,.tar.gz,.zip,.rar,.tgz')
async def disable(x):
    if not case_id.value.startswith('SR'):
        file_input.disabled = True
    else:
        file_input.disabled = False

async def change_mode(x):
    if upload_or_select.value == 'select':
        file_input.disabled = True
        case_id.disabled = True
        extract_direct.disabled = False
    else:
        case_id.disabled = False
        if not case_id.value.startswith('SR'):
            file_input.disabled = True
        else:
            file_input.disabled = False
        extract_direct.disabled = True
pn.bind(disable, case_id, watch=True)
pn.bind(change_mode, upload_or_select, watch=True)
progress = pn.indicators.Progress(active=False)
loading = pn.indicators.LoadingSpinner(width=20, height=20, value=False, visible=False)
sidebar = pn.layout.WidgetBox(
    upload_or_select,
    case_id,
    file_input,
    progress,
    pn.layout.Divider(),
    extract_direct,
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
error = pn.widgets.Tabulator(sizing_mode="stretch_both", margin=5, page_size=5, pagination='remote', show_index=False)
show_log = pn.widgets.Tabulator(styles={"font-size": "10pt"}, sizing_mode="stretch_both", margin=5, pagination=None, show_index=False)
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
show_kb = pn.widgets.Tabulator(styles={"font-size": "10pt"}, sizing_mode="stretch_both", margin=5, pagination=None, show_index=False)
kb_file, button_download_kb = show_kb.download_menu(
    text_kwargs={'name': 'Enter filename', 'value': 'kb.csv'},
    button_kwargs={'name': 'Download table'}
)
kb_tab = pn.Column(
    pn.Row(filter_file, check_kb_time, check_kb_but),
    pn.Row(pn.Column(kb_file, button_download_kb), show_kb)
)
# Check KB tab - End
# Log pattern tab - Start
filter_file_pattern = pn.widgets.Select(
    name='Choose log files to search', options=['All log', 'Only chassisd', 'Only messages'],
    value=['All log'], align='end'
)
get_data_but = pn.widgets.Button(name="Find similar ticket", align='end')
filter_time_log_pattern = pn.widgets.DatetimeRangePicker(name="Time filter", align='end')
similar_ticket_tb = pn.widgets.Tabulator(
    sizing_mode="stretch_both",
    selectable=1,
    hidden_columns=["index", "log"],
    layout="fit_data_fill",
    editors={
        "ticket": {"editable": False, "type": "string"},
        "score": {"editable": False, "type": "number"},
        "sr": {"editable": False, "type": "string"},
    },
)
source_log = pn.widgets.Tabulator(
    sizing_mode="stretch_both",
    hidden_columns=["index"],
    layout="fit_data_fill",
    header_filters={
        'filename': {'type': 'list', 'func': 'in', 'valuesLookup': True, 'sort': 'asc', 'multiselect': True},
    }
)
ticket_log = pn.widgets.Tabulator(
    sizing_mode="stretch_both",
    hidden_columns=["index"],
    layout="fit_data_fill",
    header_filters={
        'filename': {'type': 'list', 'func': 'in', 'valuesLookup': True, 'sort': 'asc', 'multiselect': True},
    }
)
possible_ticket = pn.pane.HTML("<h1>Possible ticket</h1>")
display_log_pattern = pn.GridSpec(sizing_mode='stretch_both')
display_log_pattern[0:2, 0:6] = similar_ticket_tb
display_log_pattern[0:2, 6:12] = possible_ticket
display_log_pattern[2:6, 0:6] = source_log
display_log_pattern[2:6, 6:12] = ticket_log
log_pattern_tab = pn.Column(
    pn.Row(filter_file_pattern, filter_time_log_pattern, get_data_but),
    display_log_pattern
)
# Log pattern tab - End
# Log facility & severity & event - Start
filter_file_fse_tab = pn.widgets.CheckBoxGroup(
    name='Log files', options=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    value=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    inline=True,
    align='end'
)
filter_time_fse_tab = pn.widgets.DatetimeRangePicker(name="Time filter", align='end')
parse_but = pn.widgets.Button(name="Parse syslog", align='end')
time_group = pn.widgets.Select(name='Time interval', options=['5min', '30min', '60min'], value='5min')
attr_group = pn.widgets.Select(name='Field', options=['junos_facilityname', 'junos_severitycode', 'junos_eventname'], value='junos_facilityname')
pnl = pn.Row(pn.pane.Bokeh(figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=700, tools='pan,xwheel_zoom,box_zoom,reset')))
fse_tab = pn.Column(
    pn.Row(filter_file_fse_tab, filter_time_fse_tab, parse_but),
    pn.Row(time_group, attr_group),
    pnl
)
# Log facility & severity - End
# Ticket tab - Start
def load_ticket_data():
    encoded_password = urllib.parse.quote_plus(ticket_db["password"])

    database_uri = f"mysql+pymysql://{ticket_db['user']}:{encoded_password}@{ticket_db['host']}:{ticket_db['port']}/svtech_log"
    df = pd.read_sql_query('SELECT id, tag_name, file_name, start_time, stop_time, customer, tag_optional, description FROM ticket WHERE id > 10;', database_uri)
    return df
    
tag_name = pn.widgets.TextInput(name="Tag name", placeholder="Case ID", align='end')

ticket_id = pn.widgets.TextInput(name="id", disabled=True)
ticket_file = pn.widgets.TextInput(name="File name")
ticket_time = pn.widgets.DatetimeRangePicker(name="Error time")
customer = pn.widgets.Select(name="Customer", options=['viettel', 'metfone', 'unitel', 'movitel', 'vnpt', 'mobifone', 'cmc', 'natcom', 'ftel'])
tag_optional = pn.widgets.TextInput(name="Optional tag", placeholder="Option tags e.g. KB, bgp, ospf, ...")
description = pn.widgets.TextEditor(name="Description", height=300, width= 700, value='''
<h1>Root cause</h1>
Please describe why the error occur
<h1>Impact</h1>
Please describe which component or service are affected by this error
<h1>Solution</h1>
Please describe the solution to resolve this error
''', align='end')
save_but = pn.widgets.Button(name="Update")
ticket_list = pn.widgets.Tabulator(load_ticket_data(), sizing_mode="stretch_both", selectable=1, hidden_columns=['index'],
                                   editors={
                                       'id': {'editable': False, 'type': 'number'},
                                       'tag_name': {'editable': False, 'type': 'string'},
                                       'file_name': {'editable': False, 'type': 'string'},
                                       'start_time': {'editable': False, 'type': 'string'},
                                       'stop_time': {'editable': False, 'type': 'string'},
                                       'customer': {'editable': False, 'type': 'string'},
                                       'tag_optional': {'editable': False, 'type': 'string'},
                                       'description': {'editable': False, 'type': 'string'},
                                    }, 
                                   header_filters={
                                        'tag_name':  {'type': 'input', 'func': 'like'},
                                        'file_name':  {'type': 'input', 'func': 'like'},
                                    })
ticket_tab = pn.Column(
    ticket_list,
    pn.Row(
        pn.Column(ticket_id, tag_name, ticket_file, ticket_time, customer, tag_optional),
        description,
    ),
    save_but,
    width=1500,
    align='end'
)
def click_ticket_table(event):
    ticket_id.value = str(ticket_list.value.at[event.row, 'id'])
    tag_name.value = ticket_list.value.at[event.row, 'tag_name']
    ticket_file.value = ticket_list.value.at[event.row, 'file_name']
    try:
        ticket_time.value = (ticket_list.value.at[event.row, 'start_time'], ticket_list.value.at[event.row, 'stop_time'])
    except:
        ticket_time.value = None
    if ticket_list.value.at[event.row, 'customer']:
        customer.value = ticket_list.value.at[event.row, 'customer']
    else:
        customer.value = ''
    if ticket_list.value.at[event.row, 'tag_optional']:
        tag_optional.value = ticket_list.value.at[event.row, 'tag_optional']
    else:
        tag_optional.value = ''
    if ticket_list.value.at[event.row, 'description']:
        description.value = ticket_list.value.at[event.row, 'description']
    else:
        description.value = '''
        <h1>Root cause</h1>
        Please describe why the error occur
        <h1>Impact</h1>
        Please describe which component or service are affected by this error
        <h1>Solution</h1>
        Please describe the solution to resolve this error
        '''

ticket_list.on_click(click_ticket_table)
# Ticket tab - End
# View raw log - Start
filter_file_raw_log = pn.widgets.CheckBoxGroup(
    name='Log files', options=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    value=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
    inline=True,
    align='end'
)
header_filter = {
    'filename': {'type': 'list', 'func': 'in', 'valuesLookup': True, 'sort': 'asc', 'multiselect': True},
    'time': {'type': 'input', 'func': 'like'},
    'junos_severitycode': {'type': 'list', 'func': 'in', 'valuesLookup': True, 'sort': 'asc', 'multiselect': True},
    'log': {'type': 'input', 'func': 'like'}
}
raw_log_table = pn.widgets.Tabulator(styles={"font-size": "9pt"}, layout='fit_data_table', sizing_mode="stretch_both", 
                                     min_width=800, pagination=None, show_index=False, header_filters=header_filter)
filter_time_raw_log = pn.widgets.DatetimeRangePicker(name="Time filter", align='end')
filter_rawlog_but = pn.widgets.Button(name="Get data", align='end')
raw_log_tab = pn.Column(
    pn.Row(filter_file_raw_log, filter_time_raw_log, filter_rawlog_but),
    pn.pane.Bokeh(figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=200, tools='pan,xwheel_zoom,box_zoom,reset')),
    raw_log_table
)
suggest_filter = pn.widgets.Select(name='Type', options=['chassisd*', 'config-changes*', 'interactive-commands*', 'jam_chassisd*', 'message*', 'security*'],
                                   value='message*')
suggest_file = pn.widgets.Select(name="File")
single_file_table = pn.widgets.Tabulator(styles={"font-size": "9pt"}, layout='fit_data_table', sizing_mode="stretch_both", 
                                     min_width=800, pagination=None, show_index=False, header_filters=header_filter)
single_file_tab = pn.Column(
    pn.Row(suggest_filter, suggest_file),
    pn.pane.Bokeh(figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=200, tools='pan,xwheel_zoom,box_zoom,reset')),
    single_file_table
)

def load_suggest_file(event):
    process_files = []
    process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), suggest_filter.value)
    suggest_file.options = process_files

def load_single_file_info(event):
    now = datetime.now()
    last10y = now - timedelta(weeks=520)
    process_data = syslog_rust.processing_log([suggest_file.value], last10y.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.DataFrame(process_data)[['filename', 'time', 'log']]
    single_file_table.value = df
    p = figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=200, tools='pan,xwheel_zoom,box_zoom,reset')
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M:%S",
        days="%Y-%m-%d",
        months="%Y-%m-%d",
        years="%Y-%m-%d"
    )
    p.y_range.start=0
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'], inplace=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df.set_index('time', inplace=True)
    resampled_df = df.resample('60T').count()
    p.vbar(x='time', top='log', source=ColumnDataSource(resampled_df), width=0.9)
    single_file_tab[1] = pn.pane.Bokeh(p)

suggest_filter.param.watch(load_suggest_file, 'value')
suggest_file.param.watch(load_single_file_info, 'value')
raw_tab = pn.Tabs(
    ('Single file', single_file_tab),
    ('Multiple files', raw_log_tab)
)
# View raw log - End
# Feedback tab - Start
feedback_file = pn.widgets.MultiChoice(name="File name", options=find_file(os.path.join(data_path, 'file_upload')), max_items=1)
feedback_text = pn.widgets.StaticText(name='Feedback', value='Does this app help you to localize error ?')
feedback_radio = pn.widgets.RadioBoxGroup(name="radio-check", options=['yes', 'no'], inline=True)
feedback_but = pn.widgets.Button(name="Save")
feedback_tab = pn.Column(feedback_file, feedback_text, feedback_radio, feedback_but)
# Feedback tab - End
# Search non-pattern KB and email tab - Start
input_search = pn.widgets.TextAreaInput(name='Search content', placeholder='Enter content to search here...',
                                        auto_grow=True, resizable="both", max_rows=20,
                                        sizing_mode='stretch_both')
search_button = pn.widgets.Button(name='Search', align='start')
radio_search = pn.widgets.RadioBoxGroup(name='Search method', options=['Fulltext search', 'Vector search'], inline=True)
result_display = pn.pane.HTML("<h1>Match document</h1>", styles={
    'background-color': '#F6F6F6',
    'overflow': 'auto',
    'font-size': '14px',
},sizing_mode="stretch_both")
possible_kb = pn.pane.HTML("<h1>Mentioned KB/PR in email</h1>", styles={
    'background-color': '#F6F6F6',
    'overflow': 'auto',
    'font-size': '12px',
},sizing_mode="stretch_both")
result_mapping = pn.widgets.Tabulator(sizing_mode="stretch_both", selectable=1, hidden_columns=['index', 'content'],
                                      layout='fit_data_fill', editors={
                                       'id': {'editable': False, 'type': 'string'},
                                       'score': {'editable': False, 'type': 'number'},
                                      })
search_tab = pn.GridSpec(sizing_mode='stretch_both')
search_tab[3:5, :2] = result_display
search_tab[1:3, 1] = possible_kb
full_display = pn.pane.JSON({}, sizing_mode='stretch_both', theme='light', styles={
    'background-color': '#F6F6F6',
    'overflow': 'auto',
    'font-size': '14px',
})
search_tab[:, 2:4] = full_display
search_tab[0, :2] = pn.Row(input_search, pn.Column(radio_search, search_button))
search_tab[1:3, 0] = result_mapping

async def click_result_mapping(event):
    result_display.loading = True
    possible_kb.loading = True
    result_display.object = "{}".format(result_mapping.value.at[event.row, 'content'].replace('\n', '<br>'))
    result_display.loading = False
    # Load full email from surreal
    db = SurrealHTTP('http://{}:{}'.format(surreal_db['host'], surreal_db['port']), namespace=surreal_db['namespace'], database='sr_raw',
                     username=surreal_db['user'], password=surreal_db['password'])
    try:
        res = await db.select('sr:{}'.format(result_mapping.value.at[event.row, 'id']))
        if len(res) > 0:
            kb_list = re.findall(r'KB\d+', res[0]['content'])
            pr_list = re.findall(r'PR\d+', res[0]['content'])
            possible_kb.object = """
            <h1>Mentioned KB</h1>
            {}
            <h1>Mentioned PR</h1>
            {}
            """.format("<br>".join(['<a href="https://kb.juniper.net/InfoCenter/index?page=content&id={}" target="_blank" rel="noopener noreferrer">{}</a>'.format(x, x) for x in set(kb_list).difference({'KB35593', 'KB27882', 'KB28506', 'KB21476'})]), 
                       "<br>".join(['<a href="https://prsearch.juniper.net/InfoCenter/index?page=prcontent&id={}" target="_blank" rel="noopener noreferrer">{}</a>'.format(x, x) for x in set(pr_list)]))
            full_display.object = res[0]['content'].split('\n')
    except:
        pn.state.notifications.warning("Can't load full email from database")
    await db.close()
    possible_kb.loading = False

result_mapping.on_click(click_result_mapping)
async def search_email(event):
    result_mapping.loading = True
    email_result = []
    if radio_search.value == 'Fulltext search':
        client_es = Elasticsearch([{'host': es_connection['host'], 'port': es_connection['port']}], http_auth=('elastic', 'juniper@123'), verify_certs=False, use_ssl=True)
        res = client_es.search(index="sr-svtech",  request_timeout=60, body={
            "query": {
                "match": {
                    "context": input_search.value
                }
            },
            "highlight" : {
                "pre_tags" : ["<b>"],
                "post_tags" : ["</b>"],
                "fields" : {
                "context" : {}
                }
            },
            "_source": ["SR"]
        })
        if len(res['hits']['hits']) > 0:
            for element in res['hits']['hits']:
                email_result.append({
                    "id": element['_source']['SR'],
                    "score": element['_score'],
                    "content": "\n".join(element['highlight']['context'])
                })
        else:
            pn.state.notifications.warning("Did not match any documents", duration=2000)
        client_es.close()
    elif radio_search.value == 'Vector search':
        db = Surreal("ws://{}:{}/rpc".format(surreal_db['host'], surreal_db['port']))
        await db.connect()
        await db.signin({"user": surreal_db['user'], "pass": surreal_db['password']})
        await db.use(surreal_db['namespace'], 'svtech_sr')
        query = '''
        LET $query_text = "{}"; 
        LET $query_embeddings = return http::post('http://{}:8001/encode', {{ "query": $query_text }}).embedding;
        SELECT SR, content, vector::similarity::cosine(embedded_content, $query_embeddings) AS score FROM svtech_sr WHERE embedded_content <|10|> $query_embeddings;
            '''.format(input_search.value, surreal_db['host'])
        query_result = await db.query(query)
        if len(query_result) == 3:
            for element in query_result[2]['result']:
                email_result.append({
                    "id": element['SR'],
                    "score": element['score'],
                    "content": element['content']
                })
        else:
            pn.state.notifications.warning("Some error when query db", duration=2000)
    else:
        pass
    result_mapping.value = pd.DataFrame(email_result)
    result_mapping.loading = False

search_button.on_click(search_email)
# Search non-pattern KB and email tab - End

main = pn.Tabs(
        ('View raw log', raw_tab),
        ('Check KB', kb_tab),
        ('Log facility & severity', fse_tab),
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
        ('Log pattern', log_pattern_tab),
        ('Search email', search_tab),
        ('Save ticket', ticket_tab),
        ('Feedback', feedback_tab),
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
    if re.match(r'^[\w\-.]+$', file_input.filename):
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
            # Archive(path).extractall(os.path.join(extract_path, file_name))
            load_display('on')
            if 'rar' in file_input.filename:
                subprocess.run("unar -f {} -o {} >/dev/null 2>&1".format(path, os.path.join(extract_path, file_name)), shell=True, check=False)
            elif 'zip' in file_input.filename:
                subprocess.run("unzip -o {} -d {} >/dev/null 2>&1".format(path, os.path.join(extract_path, file_name)), shell=True, check=False)
            else:
                subprocess.run("tar xf {} -C {} >/dev/null 2>&1".format(path, os.path.join(extract_path, file_name)), shell=True, check=False)
            pn.state.notifications.info('Extract file done.', duration=2000)
            process_files = []
            process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), suggest_filter.value)
            suggest_file.options = process_files
            load_display('off')            
        except:
            load_display('off')
            pn.state.notifications.error("Extract error! Check your upload file or contact admin", duration=2000)
        # Add to ticket
        try:
            cnx = mysql.connector.connect(
                host=ticket_db["host"],
                port=ticket_db["port"],
                user=ticket_db["user"],
                password=ticket_db["password"],
                database="svtech_log",
                auth_plugin='mysql_native_password'
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
            cursor.execute('''SELECT id FROM ticket WHERE tag_name = '{}';'''.format(case_id.value))
            res = cursor.fetchall()
            if len(res) == 0:
                insert_query = '''
                INSERT INTO ticket (tag_name, file_name) VALUES ('{}', '{}');
                '''.format(case_id.value, file_input.filename)
                cursor.execute(insert_query)
                cnx.commit()
            else:
                pass
            cursor.close()
            cnx.close()
            ticket_list.value = load_ticket_data()
            feedback_file.options = find_file(os.path.join(data_path, 'file_upload'))
        except Exception as e:
            pn.state.notifications.error("Can NOT create ticket due to: {}".format(e))
    else:
        pn.state.notifications.error("Invalid file name, accept only [a-zA-Z0-9.-_]", duration=2000)
    
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
    if upload_or_select.value == 'upload':
        file_name, file_extension = os.path.splitext(file_input.filename)
        return os.path.join(extract_path, file_name)
    else:
        return os.path.join(extract_path, extract_direct.value[0])
    
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
    if (testing_period.value[1] - testing_period.value[0]).days < 3:
        testing_data(get_saved_data_path(), str(testing_period.value[0]), str(testing_period.value[1]))
    else:
        pn.state.notifications.error('The testing period need to be less than 3 days')
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
        return pl.Series(pool.imap(function, column))

def check_kb_click(event):
    load_display('on')
    global pattern_dict, num_core
    process_files = []
    for filterd in filter_file.value:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    process_data = syslog_rust.processing_log(process_files, str(check_kb_time.value[0]), str(check_kb_time.value[1]))
    if len(process_data) == 0:
        pn.state.notifications.warning("There is no data in current time filter")
    else:
        try:
            list_log = list()
            for row in process_data:
                list_log.append(row['log'])
            revert_kb = {v: k for k, v in pattern_dict.items()}
            result = tag_template_rust.tag_strings(list_log, revert_kb, num_core)
            df = pd.DataFrame(process_data)
            df['kb'] = result
            df.dropna(inplace=True)
            if len(df) > 0:
                show_kb.value = df[['kb', 'time', 'filename', 'log']]
            else:
                show_kb.value = pd.DataFrame([{'result': 'there is no log match KB'}])
        except Exception as e:
            pn.state.notifications.error("Check kb error due to: {}".format(e))
    load_display('off')
check_kb_but.on_click(check_kb_click)
# Check KB tab - End
# Log pattern tab - Start
def calculate_topic_distribution(sub_df):
    global lda_model, dictionary
    sub_df["template"] = sub_df["template"].str.replace("<\*>", " ")
    sub_df["template"] = sub_df["template"].str.replace("_", " ")
    sub_df["template"] = sub_df["template"].str.replace(
        "Internal AS", "internal autonomous system"
    )
    sub_df["template"] = np.where(
        sub_df["template"].isnull(), sub_df["log"], sub_df["template"]
    )
    sub_df["process_template"] = sub_df["template"].map(preprocessing.preprocess)
    process_data = sub_df["process_template"]
    mean_distribution = [0.0] * lda_model.num_topics
    for doc in process_data:
        bow = dictionary.doc2bow(doc)
        for index, score in lda_model.get_document_topics(bow, minimum_probability=0.0):
            mean_distribution[index] += score
    sub_df["concat"] = sub_df["filename"] + "|" + sub_df["log"]
    raw = sub_df["concat"].str.cat(sep="")
    return [x / len(process_data) for x in mean_distribution], raw

async def query_vector(emb, db, table):
    query = """
    LET $query_vector = {};
    SELECT tag, vector::similarity::cosine(vector, $query_vector) AS dist, sr FROM {} WHERE vector <|1|> $query_vector;
    """.format(
        emb, table
    )
    res = await db.query(query)
    tag = None
    dist = 0
    sr = None
    try:
        tag = res[1]["result"][0]["tag"]
        dist = res[1]["result"][0]["dist"]
        sr = res[1]["result"][0]["sr"]
    except:
        pass
    return tag, dist, sr

def find_template_id(row):
    global regex_dict
    for regex_pattern, template_id in regex_dict.items():
        if re.match(regex_pattern, row):
            return template_id
    return None

async def get_data_click(event):
    await asyncio.sleep(0.25)
    load_display('on')
    # Get template from mysql
    global regex_dict, num_core
    try:
        pn.state.notifications.info("Get template data")
        conn = mysql.connector.connect(
            host=ticket_db["host"],
            port=ticket_db["port"],
            user=ticket_db["user"],
            password=ticket_db["password"],
            database="svtech_log"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT regex_pattern, template_id, template FROM template")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        regex_dict = {".*{}.*".format(pattern[0].replace("(", "\(").replace(')', '\)')): str(pattern[1]) for pattern in result }
        template_dict = {str(pattern[1]): pattern[2] for pattern in result}
    except Exception as e:
        pn.state.notifications.error("Can't get the template due to: {}".format(e))
    similar_ticket_tb.loading = True
    process_files = []
    for filterd in ["chassisd*", "jam_chassisd*", "message*", "security*"]:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    if (filter_time_log_pattern.value[1] - filter_time_log_pattern.value[0]).days < 3:
        pn.state.notifications.info("Start to collect data")
        process_data = syslog_rust.processing_log(process_files, str(filter_time_log_pattern.value[0]), str(filter_time_log_pattern.value[1]))
        if len(process_data) == 0:
            pn.state.notifications.warning("There is no data in current time filter")
        else:
            try:
                pn.state.notifications.info("Start to tag template")
                list_log = list()
                template = list()
                for row in process_data:
                    list_log.append(row["log"])
                log_with_template = tag_template_rust.tag_strings(list_log, regex_dict, num_core)
                for template_id in log_with_template:
                    if template_id:
                        template.append(template_dict[template_id])
                    else:
                        template.append(None)
                df = pd.DataFrame(process_data)
                table = "ticket_topic"
                pn.state.notifications.info("Done to tag template")
                df['template'] = template
                df["time"] = pd.to_datetime(df["time"])
                df.sort_values(by=["time"], inplace=True)
                df["time"] = df["time"].dt.tz_localize(None)
                df["log"] = df["log"].astype(str)
                if filter_file_pattern.value == 'Only chassisd':
                    df = df[df['filename'].str.contains('chassisd')]
                    table = "ticket_chassisd"
                elif filter_file_pattern.value == 'Only messages':
                    df = df[df['filename'].str.contains('message')]
                    table = "ticket_message"
                else:
                    pass
                df.set_index("time", inplace=True)
                resampled = df.resample("60T", origin="start")
                pn.state.notifications.info("Calculate topic distribution")
                s = list()
                db = SurrealHTTP('http://{}:{}'.format(surreal_db['host'], surreal_db['port']), namespace="ticket", database='ticket',
                     username=surreal_db['user'], password=surreal_db['password'])
                
                for i, (timestamp, sub_df) in enumerate(resampled):
                    if len(sub_df) > 20:
                        topic_dis, raw_text = calculate_topic_distribution(sub_df.copy(deep=True))
                        tag, dist, sr = await query_vector(topic_dis, db, table)
                        if dist > 0.95:
                            s.append(
                                {
                                    "time": "{}".format(timestamp),
                                    "log": raw_text,
                                    "ticket": tag,
                                    "score": dist,
                                    "sr": sr,
                                }
                            )
                await db.close()
                similar_ticket_tb.value = pd.DataFrame(s)
                pn.state.notifications.info("Done topic distribution")
                if len(s) > 0:
                    count_dict = dict()
                    for item in s:
                        if item['sr'] in count_dict:
                            count_dict[item['sr']] += 1
                        else:
                            count_dict[item['sr']] = 1
                    sorted_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                    possible_ticket.object = """<h1>Possible similar error</h1>
                    {}
                    <h1>Highest match: {}</h1>
                    """.format(
                        sorted_count, sorted_count[0]
                    )
                else:
                    possible_ticket.object = "<h1>Non matching ticket</h1>"
            except Exception as e:
                pn.state.notifications.error("Tag template id error due to: {}".format(e))
                print(e)
    else:
        pn.state.notifications.error('The testing period need to be less than 3 days')
    similar_ticket_tb.loading = False
    load_display('off')

def convert_log_to_table(log):
    res = []
    for line in log.split("\n"):
        token = line.split("|", 1)
        if len(token) > 1:
            res.append({'filename': token[0], 'log': token[1]})
    return res

async def click_ticket_table(event):
    db = SurrealHTTP('http://{}:{}'.format(surreal_db['host'], surreal_db['port']), namespace="ticket", database='ticket',
                     username=surreal_db['user'], password=surreal_db['password'])
    ticket_str = str(similar_ticket_tb.value.at[event.row, "ticket"])
    table = "ticket_topic"
    if filter_file_pattern.value == 'Only chassisd':
        table = "ticket_chassisd"
    elif filter_file_pattern.value == 'Only messages':
        table = "ticket_message"
    else:
        pass
    res = await db.select(
        "{}:{}".format(table, hashlib.md5(ticket_str.encode('utf-8')).hexdigest())
    )
    source_log.value = pd.DataFrame(convert_log_to_table(similar_ticket_tb.value.at[event.row, "log"]))
    ticket_log.value = pd.DataFrame(convert_log_to_table(res[0]["log"]))
    
similar_ticket_tb.on_click(click_ticket_table)
get_data_but.on_click(get_data_click)
# Log pattern tab - End
# Facility & severity & event tab - Start
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
    '0': 'emerg',
    '1': 'alert',
    '2': 'crit',
    '3': 'err',
    '4': 'warning',
    '5': 'notice',
    '6': 'info',
    '7': 'debug'
}

def parse_pygrok(line):
    global grok, reversed_severity
    res = grok.match(line['log'])
    if res is not None:
        line['junos_facilityname'] = res['junos_facilityname']
        line['junos_severitycode'] = reversed_severity[res['junos_severitycode']]
        line['junos_eventname'] = res['junos_eventname']

def create_figure():
    global parse_data
    p = figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=700, tools='pan,xwheel_zoom,box_zoom,reset')
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M:%S",
        days="%Y-%m-%d",
        months="%Y-%m-%d",
        years="%Y-%m-%d"
    )
    p.y_range.start=0
    if parse_data is not None:
        df = pd.DataFrame(parse_data)
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by=['time'], inplace=True)
        df['time'] = df['time'].dt.tz_localize(None)
        df.set_index('time', inplace=True)
        if attr_group.value in df.columns: 
            count = df.resample(time_group.value)[attr_group.value].value_counts().unstack(fill_value=0)
            count = count.asfreq(time_group.value, fill_value=0)
            source = ColumnDataSource(count)
            colors = Category20[20]
            legend = Legend(items=[(column, [p.line(x='time', y=column, source=source, line_width=2, line_color=colors[i])]) for i, column in enumerate(count.columns) if i<20])
            # Add the Legend to the plot
            p.add_layout(legend, 'right')
            p.legend.click_policy = 'hide'
            # Customize the plot
            p.xaxis.axis_label = 'Time'
            p.yaxis.axis_label = 'Count'
        else:
            pn.state.notifications.error("There is no {} filter, please choose other".format(attr_group.value))
    else:
        pass
    return pn.pane.Bokeh(p)

def get_data_and_parse(event):
    global parse_data
    load_display('on')
    process_files = []
    for filterd in filter_file_fse_tab.value:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    process_data = syslog_rust.processing_log(process_files, str(filter_time_fse_tab.value[0]), str(filter_time_fse_tab.value[1]))
    if len(process_data) == 0:
        parse_data = None
        pn.state.notifications.warning("There is no data in current time filter")
    else:
        rfc5424 = False
        for ele in process_data[0:5]:
            if ele['log'].startswith("<"):
                rfc5424 = True
        if rfc5424:
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(parse_single_line_rfc5424, process_data)
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(parse_pygrok, process_data)
    parse_data = process_data
    pnl[0] = create_figure()
    load_display('off')

def replace_plot(event):
    pnl[0] = create_figure()
    
parse_but.on_click(get_data_and_parse)
time_group.param.watch(replace_plot, 'value')
attr_group.param.watch(replace_plot, 'value')
# Facility & severity & event tab - End
# View raw log - Start
def filter_raw_log(event):
    load_display('on')
    process_files = []
    for filterd in filter_file_raw_log.value:
        process_files += BASE_LOG_ANALYSE.get_file_list_by_filename_filter(get_saved_data_path(), filterd)
    process_data = syslog_rust.processing_log(process_files, str(filter_time_raw_log.value[0]), str(filter_time_raw_log.value[1]))
    if len(process_data) == 0:
        pn.state.notifications.warning("There is no data in current time filter")
    else:
        rfc5424 = False
        for ele in process_data[0:5]:
            if ele['log'].startswith("<"):
                rfc5424 = True
        if rfc5424:
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(parse_single_line_rfc5424, process_data)
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(parse_pygrok, process_data)
        df = pd.DataFrame(process_data)
        if 'junos_severitycode' in df.columns:
            df = df[['filename', 'time', 'junos_severitycode', 'log']]
            df.fillna('', inplace=True)
        else:
            df = df[['filename', 'time', 'log']]
        pn.state.notifications.info("Number of logs: {}".format(len(df)), duration=5000)
        raw_log_table.value = df
        p = figure(x_axis_type='datetime', title='Count vs. Time', width=1400, height=200, tools='pan,xwheel_zoom,box_zoom,reset')
        p.xaxis.formatter = DatetimeTickFormatter(
            hours="%H:%M:%S",
            days="%Y-%m-%d",
            months="%Y-%m-%d",
            years="%Y-%m-%d"
        )
        p.y_range.start=0
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by=['time'], inplace=True)
        df['time'] = df['time'].dt.tz_localize(None)
        resampled_df = df.copy(deep=True).set_index('time')
        resampled_df = resampled_df.resample('60T').count()
        p.vbar(x='time', top='log', source=ColumnDataSource(resampled_df), width=0.9)
        raw_log_tab[1] = pn.pane.Bokeh(p)
    load_display('off')
        
filter_rawlog_but.on_click(filter_raw_log)
# View raw log - End
# Save ticket - Start
def save_but_click(event):
    try:
        cnx = mysql.connector.connect(
            host=ticket_db["host"],
            port=ticket_db["port"],
            user=ticket_db["user"],
            password=ticket_db["password"],
            database="svtech_log",
            auth_plugin='mysql_native_password'
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
        update_query = '''
        UPDATE ticket 
        SET file_name = '{}', start_time = '{}', stop_time = '{}', customer = '{}', tag_optional = '{}', description = '{}', tag_name = '{}'
        WHERE id = {};
        '''.format(ticket_file.value, str(ticket_time.value[0]), str(ticket_time.value[1]), customer.value, tag_optional.value, description.value, tag_name.value, ticket_id.value)
        cursor.execute(update_query)
        cnx.commit()
        cursor.close()
        cnx.close()
        pn.state.notifications.info("Update succesfully!")
        ticket_list.value = load_ticket_data()
    except Exception as e:
        pn.state.notifications.error("Can NOT update ticket due to: {}".format(e))

save_but.on_click(save_but_click)
# Save ticket - End
# Feedback - Start
def save_feedback_but(event):
    try:
        cnx = mysql.connector.connect(
            host=ticket_db["host"],
            port=ticket_db["port"],
            user=ticket_db["user"],
            password=ticket_db["password"],
            database="svtech_log"
        )
        cursor = cnx.cursor()
        table_name = 'feedback'
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()
        if not result:
            table_creation_query = f'''
            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `file_name` text DEFAULT NULL,
                `feedback` text DEFAULT NULL
            )
            '''
            cursor.execute(table_creation_query)
        insert_query = '''
        INSERT INTO feedback (file_name, feedback)
        VALUES ('{}', '{}');
        '''.format(feedback_file.value[0], feedback_radio.value)
        cursor.execute(insert_query)
        cnx.commit()
        cursor.close()
        cnx.close()
        pn.state.notifications.info("Save succesfully!")
    except Exception as e:
        pn.state.notifications.error("Can NOT save feedback due to: {}".format(e))
feedback_but.on_click(save_feedback_but)
# Feedback - End
# Component logic - End

# ROUTES = {
#     "app_panel": main_page,
# }
# pn.serve(ROUTES,
#          port=3001,
#          address='0.0.0.0',
#          websocket_origin='*',
#          websocket_max_message_size=MAX_SIZE_MB*1024*1024,
#          http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1024}
#         )
main_page.servable()
file_input.disabled=True
extract_direct.disabled = True