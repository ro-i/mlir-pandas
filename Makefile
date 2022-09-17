.ONESHELL:

db_dir = ~/dev/tum/BA/lingo-db
db_build_dir = $(db_dir)/build/lingodb-release
ld_library_path = $(db_dir)/build/arrow/install/lib
pythonpath = $(db_dir)/arrow/python/:$(db_build_dir)/
mlir_pandas_dir = $(shell pwd)
benchmark_dir = benchmarks
dbgen_dir = ~/dev/tum/BA/tpch-dbgen

.PHONY: run-test
run-test:
	@. venv/bin/activate
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath) \
			python3 ./test.py

.PHONY: benchmarks
benchmarks:
	@. venv/bin/activate
	cd $(benchmark_dir)
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath):./mlir_pandas python3 ./benchmark.py

.PHONY: benchmarks-debug
benchmarks-debug:
	@. venv/bin/activate
	cd $(benchmark_dir)
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath):./mlir_pandas python3 ./benchmark.py -d

.PHONY: benchmarks-test
benchmarks-test:
	@. venv/bin/activate
	cd $(benchmark_dir)
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath):./mlir_pandas python3 ./test_csv.py

# Run tools/generate/tpch.sh in lingo-db repository to generate appropriate data.
.PHONY: tpch_data
tpch_data:
	rm -f $(benchmark_dir)/tpch_data
	ln -s $(db_dir)/resources/data/tpch-1 $(benchmark_dir)/tpch_data

.PHONY: tpch_data_lingodb
tpch_data_lingodb:
	rm -f $(benchmark_dir)/tpch_data
	ln -s $(db_dir)/resources/data/tpch $(benchmark_dir)/tpch_data

.PHONY: mypy
mypy:
	@. venv/bin/activate
	mypy --strict \
		--allow-subclassing-any \
		--allow-untyped-decorators \
		--ignore-missing-imports \
		--no-warn-return-any \
		--no-warn-unused-ignores \
		mlir_pandas tests benchmarks
.PHONY: shell
shell:
	@. venv/bin/activate
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath) python3

.PHONY: _tests
_tests:
	@. venv/bin/activate
	LD_LIBRARY_PATH=$(ld_library_path) PYTHONPATH=$(pythonpath) pytest

.PHONY: venv
venv:
	@test -e venv || python3 -m venv venv
	. venv/bin/activate
	pip3 install -r requirements.txt -U
