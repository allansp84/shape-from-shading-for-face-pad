#!/bin/bash

for req in $(cat requirements.txt); do
    pip install $req;
done