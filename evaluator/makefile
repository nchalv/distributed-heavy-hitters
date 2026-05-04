.PHONY: all clean build release debug test
MAKEFLAGS += --no-print-directory

# Default target
all: release

# Your exact compile commands
release:
	@echo "Building Release..."
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j
	@echo "Done Building..."

debug:
	@echo "Building Debug..."
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
	@cmake --build build -j
	@echo "Done Building..."

# Clean everything
clean:
	@echo "Cleaning project..."
	@rm -rf build/
	@rm -f hh_run hh_bench hh_tests
	@find . -name "*.o" -delete 2>/dev/null || true
	@find . -name "cmake-build-*" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Done..."
# Clean and rebuild
rebuild: clean release

# Run targets (adjust as needed)
run: release
	./build/hh_run

bench: release
	./build/hh_bench

test: debug
	@ctest --test-dir build --output-on-failure

# Help target
help:
	@echo "Available targets:"
	@echo "  all, release - Build release version (default)"
	@echo "  debug        - Build debug version"
	@echo "  clean        - Remove all build files"
	@echo "  rebuild      - Clean and rebuild release"
	@echo "  run          - Build and run hh_run"
	@echo "  bench        - Build and run hh_bench"
	@echo "  test         - Build and run tests"
	@echo "  help         - Show this help"
