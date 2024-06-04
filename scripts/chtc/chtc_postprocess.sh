#!/bin/bash
# chtc_postprocess.sh

# run this on sjmc

# mv /staging/syang662/scripts_results.tar.gz /home/syang662/llm-hpv-misinfo
# scripts/Shanglin/Varsha/git_pull.sh

scp syang662@ap2002.chtc.wisc.edu:/home/syang662/llm-hpv-misinfo/scripts_results.tar.gz .

tar -xzvf scripts_results.tar.gz
# tar --overwrite -xzvf scripts_results.tar.gz