from reader import Reader

batch_size = 128

reader = Reader(20000, 40)

train_x, train_y, seq_len = next(reader.train_batch(128))

print('training data:')
print('X shape:', train_x.shape)
print('Y shape:', train_y.shape)
print('Seq shape:', seq_len.shape)
print(train_x[:3])
print(train_y[:3])
print(seq_len[:3])


test_x, test_y, seq_len = next(reader.test_batch(1040))

print('testing data:')
print('X shape:', test_x.shape)
print('Y shape:', test_y.shape)
print('Seq shape:', seq_len.shape)
print(test_x[:3])
print(test_y[:3])
print(seq_len[:3])
