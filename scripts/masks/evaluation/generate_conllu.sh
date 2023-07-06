#!/bin/bash

if [ $# -eq 1 ]; then
    read -p "" sentences
    request="curl --data 'model=$1&tokenizer=&tagger=&parser=&data=$sentences' http://lindat.mff.cuni.cz/services/udpipe/api/process"
    eval $request | PYTHONIOENCODING=utf-8 python -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])"
else
    mkdir -p /tmp/conllu-tmp-process/
    split -d -l 10000 $2 /tmp/conllu-tmp-process/conllu-files

    for file in /tmp/conllu-tmp-process/*
    do
        curl -F data=@$file -F model=$1 -F input=horizontal -F tagger= \
            -F parser= http://lindat.mff.cuni.cz/services/udpipe/api/process | PYTHONIOENCODING=utf-8 python -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])"
    done
fi
