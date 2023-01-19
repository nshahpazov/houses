preprocess:
	@echo "Preprocessing the data for the houses project"
	@mkdir -p data/interim data/proccessed
	@python ./src/preprocess.py data/raw/train.csv data/interim/train.csv
	@python ./src/preprocess.py data/raw/test.csv data/interim/test.csv
