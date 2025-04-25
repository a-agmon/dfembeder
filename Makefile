.PHONY: build install test all clean

# Build the Rust project
build:
	cargo build

# Install the Python module
install: build
	pip install -e .

# Run the Python tests
test: install
	pytest -s -v tests/test_integration.py

# Run the example script
run: install
	python python_tests/test_analyzer.py

# Build, install, and test
all: build install test

# Clean build artifacts
clean:
	cargo clean
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	find . -name "__pycache__" -type d -exec rm -rf {} +