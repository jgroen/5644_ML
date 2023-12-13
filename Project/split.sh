#!/bin/bash

for file in ./project_data/*.csv
do
  sed -i '8503q' ${file}
done 
