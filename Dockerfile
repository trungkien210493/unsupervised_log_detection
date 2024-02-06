FROM python:3.9.13
RUN apt-get update && apt-get -qq --no-install-recommends -y install vim lftp wget gcc python3-dev libc-dev build-essential graphviz unzip unrar-free p7zip-full
WORKDIR /unsupervised_log_detection
COPY ./requirement.txt /tmp/requirement.txt
RUN pip install --no-cache-dir -r /tmp/requirement.txt
COPY . .
RUN pip install /unsupervised_log_detection/syslog_rust-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl rich
EXPOSE 3001
CMD ["python", "/unsupervised_log_detection/app_panel.py"]
