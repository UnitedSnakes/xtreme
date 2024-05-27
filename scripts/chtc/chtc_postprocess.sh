#!/bin/bash
# chtc_postprocess.sh

# run this on sjmc

# mv /staging/syang662/content_results.tar.gz /home/syang662/llm-hpv-misinfo
# scripts/Shanglin/Varsha/git_pull.sh

scp syang662@ap2002.chtc.wisc.edu:/home/syang662/llm-hpv-misinfo/content_results.tar.gz .

tar -xzvf content_results.tar.gz
# tar --overwrite -xzvf content_results.tar.gz