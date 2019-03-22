FROM tensorflow/tensorflow:latest-gpu-py3
# set the working directory
RUN mkdir app
WORKDIR app


# install code dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# install environment dependencies
COPY run.sh .
COPY train.py .
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# provision environment
RUN chmod +x ./run.sh
EXPOSE 8080
ENTRYPOINT ["./run.sh"]
CMD ["build"]