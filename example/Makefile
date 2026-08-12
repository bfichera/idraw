.PHONY: env run clean dev

env:
	python -m venv workflow/envs/venv && source workflow/envs/venv/bin/activate && pip install -r requirements.txt

dev:
	python -m venv runtime-env && source runtime-env/bin/activate && pip install snakemake

run:
	source runtime-env/bin/activate && snakemake --profile profiles/local

clean:
	rm -rf .snakemake/ && rm -rf assets/ && rm -rf plots/ && rm -rf runtime-env/ && rm -rf workflow/envs/venv
