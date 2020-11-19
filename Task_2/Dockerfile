FROM continuumio/anaconda3:4.4.0
COPY . /usr/app2/
EXPOSE 5000
WORKDIR /usr/app2/
RUN pip install -r requirements.txt
CMD python model.py