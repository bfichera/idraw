.PHONY: env run clean

env:
	python -m venv workflow/envs/venv && source workflow/envs/venv/bin/activate && pip install -r requirements.txt

run:
	snakemake --profile profiles/local

clean:
	rm -rf .snakemake/ && rm -rf assets/ && rm -rf plots/
