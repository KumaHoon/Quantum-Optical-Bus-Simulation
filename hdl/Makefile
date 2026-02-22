SHELL := /bin/sh
PYTHON := python
IVERILOG := iverilog
VVP := vvp
RTL := feedforward_lut.sv
TB := tb_feedforward_lut.sv
OUTPUT := sim.out
VCD := waves.vcd
VECTORS := vectors/stimulus.mem vectors/expected.mem
VEC_GEN := ../scripts/export_golden_vectors.py

ifeq ($(OS),Windows_NT)
RM := del /F /Q
else
RM := rm -f
endif

all: sim

$(VECTORS): $(VEC_GEN)
	$(PYTHON) $(VEC_GEN) --output-dir vectors --count 128 --seed 13 --manifest vectors/contract.json

sim: $(VECTORS)
	$(IVERILOG) -g2012 -o $(OUTPUT) $(RTL) $(TB)
	$(VVP) $(OUTPUT)

lint:
	@if ! command -v $(IVERILOG) >/dev/null 2>&1; then \
		echo "iverilog not found. Install Icarus Verilog."; \
		exit 1; \
	fi
	$(IVERILOG) -g2012 -t null -Wall -s tb_feedforward_lut $(RTL) $(TB)

clean:
	$(RM) $(OUTPUT) $(VCD)

.PHONY: all sim lint clean
