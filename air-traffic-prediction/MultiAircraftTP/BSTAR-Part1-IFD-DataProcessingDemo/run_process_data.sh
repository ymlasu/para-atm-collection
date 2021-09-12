#!/bin/bash

for r in 0.1
do
  for d in 20190801 20190802 20190803 20190804 20190805 20190806 20190807
  do
    for a in 8
    do
      python data_parser.py --range $r --duration $a --date $d
    done
  done
done
