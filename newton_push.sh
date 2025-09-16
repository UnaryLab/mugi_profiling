#!/bin/bash
set -e  # stop if any command fails

git add .
git commit -m "push to newton cluster"
git push
