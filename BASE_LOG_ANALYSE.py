import glob
import gzip
import os
import re
import tarfile
import logging
import time
import ast
from pprint import pprint
from io import open
import fnmatch
import zlib
from multiprocessing import Process, Queue, Manager
num_threads = 3


def parse_list(args):
    args = ast.literal_eval(args)
    if type(args) is not list:
        raise Exception("The type of replaced list [ {} ] is not LIST".format(args))
    return args


def get_file_list_by_filename_filter(directory, filter):
    if directory[-1] != "/":
        directory = directory + "/"
    for root, dirname, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, filter):
            yield os.path.join(root, filename)
    #return [f for f in glob.glob(directory + "**/*{}*".format(filter), recursive=True)]


def extract_all_tar_file_in_folder(directory):
    if directory[-1] != "/":
        directory = directory + "/"
    list_tar_file = glob.glob(directory + "/*.tgz")
    for file in list_tar_file:
        logging.info("Working on file {}".format(file))
        file_name = os.path.basename(file)
        file_name = file_name.replace('.tgz', '')
        if not os.path.exists(directory + file_name):
            os.makedirs(directory + file_name)
        tar = tarfile.open(file)
        tar.extractall(path=directory + file_name)
        tar.close()


def read_gzip_file(file_name, chunk_size=1024):
    content = ''
    try:
        with gzip.open(file_name, 'rb') as f_in:
            content = f_in.read().decode("latin-1")
    except Exception as e:
        logging.warning("Can't read all data in '{}' due to {}".format(file_name, e))
        d = zlib.decompressobj(zlib.MAX_WBITS | 32)
        with open(file_name, 'rb') as f:
            buffer = f.read(chunk_size)
            try:
                while buffer:
                    content += d.decompress(buffer).decode("latin-1")
                    buffer = f.read(chunk_size)
            except Exception:
                # By pass the corrupted data in file
                pass
    return content


#def read_xslx_file(xlsx_file):
#    book = openpyxl.load_workbook(xlsx_file)
#    sheet = book.get_sheet_by_name("MPC Error type")
#    columns_name = []
#    for cell in sheet[1]:
#        columns_name.append(cell.value)
#    index_regex = columns_name.index("REGEX (Matching)") + 1
#    index_ID = columns_name.index("Juniper KB") + 1
#    result_dict = {}
#    max_row = sheet.max_row
#    for i in range(2, max_row+1):
#        if sheet.cell(row=i, column=index_regex).value != '':
#            juniper_KB = sheet.cell(row=i, column=index_ID).value
#            regex = sheet.cell(row=i, column=index_regex).value
#            if juniper_KB:
#                juniper_KB = juniper_KB.replace("\n", '')
#            if regex:
#                regex = regex.replace("\n", '')
#            result_dict[juniper_KB] = regex
#    result_dict.pop(None, None)
#    return result_dict


def parsing_line(line):
    tokens = line.split()
    pri_code = ''
    if len(tokens) < 4:
        timestamp = 'Unknown'
        log_info = line
    else:
        month = tokens[0]
        day = tokens[1]
        time = tokens[2]
        if tokens[3].isdigit():
            year = tokens[3]
            log_info = ' '.join(tokens[4:])
        else:
            year = ''
            log_info = ' '.join(tokens[3:])
        timestamp = month + ' ' + day + ' ' + time + ' ' + year

        ### parsing for lines with pri-code ###
        if "<" in tokens[0] and "T" in tokens[1]:
            pri_code = tokens[0]
            string_time = tokens[1]
            string_time = string_time.split("T")
            year_month_day = string_time[0].split("-")
            year = year_month_day[0]
            month = year_month_day[1]
            day  = year_month_day[2]
            time = string_time[1]
            timestamp = month + ' ' + day + ' ' + time + ' ' + year
            log_info = ' '.join(tokens[4:])
        ##
    return [timestamp, log_info, pri_code]

def analysing_log_multi_process(queue_index, file_queue, line_regex_found, host_name, pattern_dict):
    while True:
        if file_queue.empty():
            break
        file_name = file_queue.get()
        if file_name is None:
            file_queue.put(None)
            break
        # Read file content - Start
        if '.gz' in file_name:
            logging.info("Unzipping file {} for reading".format(file_name))
            content = read_gzip_file(file_name)
        else:
            logging.info("Reading raw file {}".format(file_name))
            with open(file_name, 'r', encoding="latin-1") as f_in:
                content = f_in.read()
        # Read file content - End
        reduced_pattern = {}
        logging.info('BEGIN: Looking for pattern existence in {} before regexing each line'.format(file_name))
        for kb, regex in pattern_dict.items():
            res = re.findall(r"{}".format(regex), content)
            if len(res) > 0:
                reduced_pattern[kb] = regex
        # Reduced pattern - End
        if reduced_pattern: #only start analyzing line by line if some pattern is found in file
            logging.warning('RESULT: Some pattern was found in file {}, they are {}, putting in queue index {}'.format(file_name, reduced_pattern,queue_index))
            result = analysing_log(host_name, file_name, reduced_pattern, content.split("\n"))
            if len(result) > 0:
                line_regex_found += result
        else:
            logging.info('RESULT: No pattern was detected in file {}, SKIPPING it'.format(file_name))
            pass



def analysing_log(host_name, file_name, pattern_dict, content):
    file_name = os.path.basename(file_name)
    line_regex_found = []
    logging.info("Working line by line on file {} of host {}".format(file_name,host_name))
    # line_regex_not_found = []
    i = 0
    for line in content:
        i += 1
        error_ID = []
        [timestamp, log_info, pri_code] = parsing_line(line)
        for key, value in pattern_dict.items():
            pattern = re.compile(r".*{}.*".format(value))
            if pattern.match(line):
                logging.debug("{}: Pattern of KB {} found in file {}, line {}".format(host_name,key, file_name, timestamp))
                error_ID.append(key)
        if len(error_ID) == 0:
            # line_regex_not_found.append([timestamp, log_info, host_name, i])
            pass
        else:
            line_regex_found.append([timestamp, log_info, host_name, file_name, i, error_ID, pri_code])
    logging.warning('line_regex_found {}'.format(line_regex_found))
    return line_regex_found


def analysing_log_single_process(file_name, host_name, pattern_dict):
    line_regex_found = []
    logging.debug('Reading content from log file: {}'.format(file_name))
    # Read file content - Start
    if '.gz' in file_name:
        content = read_gzip_file(file_name)
    else:
        with open(file_name, 'r', encoding="latin-1") as f_in:
            content = f_in.read()
    # Read file content - End

    # Reduced pattern - Start
    reduced_pattern = {}
    logging.info('BEGIN: Looking for pattern existence before regexing each line')
    for kb, regex in pattern_dict.items():
        res = re.findall(r"{}".format(regex), content)
        if len(res) > 0:
            reduced_pattern[kb] = regex
    # Reduced pattern - End
    if reduced_pattern:  # only start analyzing line by line if some pattern is found in file
        logging.warning('RESULT: Some pattern was found in file {}, they are {}'.format(file_name, reduced_pattern))
        result = analysing_log(host_name, file_name, reduced_pattern, content.split("\n"))
        if len(result) > 0:
            line_regex_found += result
    else:
        logging.info('RESULT: No pattern was detected in file {}, SKIPPING it'.format(file_name))
        pass
    return line_regex_found


def analysing_log_multiprocess_main(host_name, list_file, pattern_dict):
    file_queue = Queue()
    manager = Manager()
    line_regex_found = manager.list()
    global num_threads
    processes = []
    for queue_index in range(num_threads):
        worker = Process(target=analysing_log_multi_process,
                        args=(queue_index, file_queue, line_regex_found, host_name, pattern_dict))
        processes.append(worker)
    logging.debug("Working on file list [{}] with queue size {}".format(list_file, num_threads))
    for file in list_file:
        file_queue.put(file)
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
    return line_regex_found


def write_result_to_csv(file_name,line_regex_found):
    logging.warning('line regex write to csv {}'.format(line_regex_found))
    try:
        with open(file_name, 'w+', encoding='utf-8') as f:
            logging.info("Writing result to {}".format(file_name))
            if len(line_regex_found) == 0:
                msg = 'All syslog file is OK, no error KB matched'
                logging.info(msg)
                f.write(u'{}'.format(msg))
            else:
                f.write(u'Time,Log info,Host name,File name,Line in log file,Juniper KB,Pri code\n')
                for i in range(len(line_regex_found)):
                    logging.warning('line regex write to csv {}'.format(line_regex_found[i][6]))
                    f.write(u'{},{},{},{},{},{},{}\n'.format(
                        line_regex_found[i][0].replace(',', '___'),
                        line_regex_found[i][1].replace(',', '___'),
                        line_regex_found[i][2].replace(',', '___'),
                        line_regex_found[i][3],
                        line_regex_found[i][4],
                        '\t'.join(line_regex_found[i][5]).replace(',', '___'),
                        line_regex_found[i][6],
                    ))
    except Exception as e:
        logging.exception("Error during writing result to {} ")
        raise
