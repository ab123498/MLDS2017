wget https://www.dropbox.com/s/x7dzlgu5y71l3l7/att_model.tar.gz?dl=0 -O att_model.tar.gz

tar zxvf att_model.tar.gz

python3 att_lstm/test.py $1 $2
