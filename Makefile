LABEL_FILE=/home/amit/PycharmProjects/eNable_Narrative_Extraction/data/labelled_data/Labels.csv
GLOVE_EMBEDDING_FILE=/home/amit/PycharmProjects/eNable_Narrative_Extraction/data/glove.6B.100d.txt
OUTPUT_FILE=output.txt

all: data/train.csv output.txt

data/train.csv:
	python Split_data.py ${LABEL_FILE}

output.txt: data/train.csv
	python Classification_using_seq2seq.py ${GLOVE_EMBEDDING_FILE} >> ${OUTPUT_FILE}