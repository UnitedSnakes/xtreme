#!/bin/bash

scripts/other/pack_large_csv.sh

git add .gitignore
git commit -m "Update .gitignore"
git push

git add .

echo "Enter commit message:"
read commit_message

git commit -m "$commit_message"

git push
