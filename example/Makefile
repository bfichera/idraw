.PHONY: env run clean dev fetch-data

DATA_URL = https://github.com/bfichera/idraw/releases/download/v1.0.0/example-data.tar.gz

env:
	python -m venv workflow/envs/venv && source workflow/envs/venv/bin/activate && pip install -r requirements.txt

dev:
	python -m venv runtime-env && source runtime-env/bin/activate && pip install snakemake

fetch-data:
	mkdir -p data && curl -L $(DATA_URL) | tar -xz -C data/
	

run:
	source runtime-env/bin/activate && snakemake --profile profiles/local

clean:
	rm -rf .snakemake/ && rm -rf assets/ && rm -rf plots/ && rm -rf runtime-env/ && rm -rf workflow/envs/venv
