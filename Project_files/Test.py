from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

for _ in range(86000):
    model.add(Dense(1000, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam')

