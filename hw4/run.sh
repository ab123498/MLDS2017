if [ $1 == "S2S" ]
then
    wget https://www.dropbox.com/s/k2ibqgcxhn4b56q/s2s_model.zip?dl=0 -O s2s_model.zip

    unzip s2s_model.zip
    python3 seq2seq/test.py $2 $3
fi

if [ $1 == "RL" ]
then
    python3 RL/test.py $2 $3
fi

if [ $1 == "BEST" ]
then
    wget https://www.dropbox.com/s/k2ibqgcxhn4b56q/s2s_model.zip?dl=0 -O s2s_model.zip

    unzip s2s_model.zip
    python3 seq2seq/test.py $2 $3
fi