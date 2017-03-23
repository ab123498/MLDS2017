echo "Download trained model..."
cd basic_lstm
wget https://www.dropbox.com/s/scyhbdacqq0q50o/trained_model.tgz?dl=0 -O trained_model.tgz
tar zxvf trained_model.tgz
cd ..

echo "predicting..."
python3 basic_lstm/eval.py basic_lstm/trained_model $1 $2
