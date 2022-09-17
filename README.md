## "mlir pandas"

Combine the power of a DBMS with the usability and integration of pandas!

### Implementation
- Configure the Makefile: Set the variable `db_dir` in `Makefile` to the
  directory of your local installation of
  [lingo-db](https://github.com/lingo-db/lingo-db).
- Create / update the virtual environment: `make venv`
- Run mypy: `make mypy`
- Run benchmarks: `make benchmarks`

#### Debugging
- Drop some code in `test.py` and run it using `make run-test` or simply `make`.
- Run `make shell` to access a python "shell" called inside the virtual
  environment and with the suitable environment variables.
