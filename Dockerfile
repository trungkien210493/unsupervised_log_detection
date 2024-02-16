FROM python:3.9.13
RUN apt-get update && apt-get -qq --no-install-recommends -y install vim lftp wget gcc python3-dev libc-dev build-essential graphviz unzip unrar-free p7zip-full
WORKDIR /unsupervised_log_detection
COPY ./requirement.txt /tmp/requirement.txt
RUN pip install --no-cache-dir -r /tmp/requirement.txt
COPY . .
RUN pip install /unsupervised_log_detection/syslog_rust-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl rich
RUN sed -i 's/self\.max_buffer_size = max_buffer_size or 104857600/self\.max_buffer_size = max_buffer_size or 209715200/' /usr/local/lib/python3.9/site-packages/tornado/iostream.py
EXPOSE 3001
CMD ["panel", "serve", "/unsupervised_log_detection/app_panel.py", "--address", "0.0.0.0", "--port", "3001", "--websocket-max-message-size", "209715200", "--allow-websocket-origin","*"]
