#!/bin/bash

ssh kalkulator2 'cd LSTM-New-Look && tar zcvf results.tar.gz results/11/'
scp kalkulator2:~/LSTM-New-Look/results.tar.gz .
tar zxvf results.tar.gz

