.PHONY: install install-dev sync lock lint format format-check typecheck test coverage precommit info show-config validate-config create-dirs check-environment environment-report init-run apptainer-build apptainer-test cluster-bootstrap cluster-env-check clean
install: ; uv sync
install-dev: ; uv sync --group dev
sync: ; uv sync
lock: ; uv lock
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
typecheck: ; uv run mypy src
test: ; uv run pytest
coverage: ; uv run pytest --cov-report=term-missing
precommit: ; uv run pre-commit run --all-files
info: ; uv run fedbrats info
show-config: ; uv run fedbrats show-config
validate-config: ; uv run fedbrats validate-config
create-dirs: ; uv run fedbrats create-dirs
check-environment: ; uv run fedbrats check-environment
environment-report: ; uv run fedbrats write-environment-report --path outputs/environment.json
init-run: ; uv run fedbrats init-run
apptainer-build: ; ./apptainer/build.sh
apptainer-test: ; ./apptainer/test.sh
cluster-bootstrap: ; ./scripts/bootstrap_cluster.sh
cluster-env-check: ; ./scripts/validate_environment.sh
clean: ; rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
