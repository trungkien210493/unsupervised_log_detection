FROM python:3.9.13
RUN apt-get update && \
    apt-get -qq --no-install-recommends -y install vim lftp wget gcc python3-dev libc-dev build-essential graphviz
WORKDIR /unsupervised_log_detection
COPY ./requirement.txt /tmp/requirement.txt
RUN pip install -r /tmp/requirement.txt
COPY . .
EXPOSE 3001
CMD ["panel", "serve", "/unsupervised_log_detection/app_panel.py", "--address", "0.0.0.0", "--port", "3001", "--allow-websocket-origin","*"]