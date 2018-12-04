#!/bin/bash

ssh kalkulator2 'cd LSTM-Music-Generation && tar zcvf results.tar.gz results/'
scp kalkulator2:~/LSTM-Music-Generation/results.tar.gz .
tar zxvf results.tar.gz

