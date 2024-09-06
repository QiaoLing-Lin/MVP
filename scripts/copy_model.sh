#!/bin/sh
#
if [ ! -d log_$2 ]; then
cp -r log_ucm/log_$1 log_ucm/log_$2
cd log_ucm/log_$2
mv infos_$1-best.pkl infos_$2-best.pkl 
mv infos_$1.pkl infos_$2.pkl 
cd ../
fi