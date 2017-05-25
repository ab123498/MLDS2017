mkdir skipthoughts_data
cd skipthoughts_data

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

cd ..

wget https://www.dropbox.com/s/n3pr8iy88bnapmg/trained_model.zip?dl=0 -O trained_model.zip

unzip trained_model.zip

python3 generate.py $1