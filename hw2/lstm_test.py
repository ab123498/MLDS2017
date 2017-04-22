import os
import argparse
from lstm import test
# parser = argparse.ArgumentParser()
# parser.add_argument('testing_list')
# parser.add_argument('feat_path')
# option = parser.parse_args()
testing_dir = './MLDS_hw2_data/testing_data/feat'
testing_list = './MLDS_hw2_data/testing_id.txt'


test(testing_dir, testing_list)