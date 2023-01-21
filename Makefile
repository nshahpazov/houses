preprocess:
	@echo "Preprocessing the data for the houses project"
	@mkdir -p data/interim data/proccessed
	@python ./houses/preprocess.py data/raw/train.csv data/interim/train.csv
	@python ./houses/preprocess.py data/raw/test.csv data/interim/test.csv
