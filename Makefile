PROJECTS := $(notdir $(wildcard projects/*))
PYTHON   ?= python3

.DEFAULT_GOAL := help
.PHONY: help $(PROJECTS)

help:
	@echo "Available projects: $(PROJECTS)"
	@echo "Usage: make <project_name>     # rebuild every output for that project"

$(PROJECTS):
	@echo "==> [$@] 1/5  generate shelf outlines"
	@if [ -f projects/$@/scripts/generate.py ]; then \
		$(PYTHON) projects/$@/scripts/generate.py; \
	else echo "    (skipped — projects/$@/scripts/generate.py not present)"; fi
	@echo "==> [$@] 2/5  overlay brackets (visualization DXF + PDF)"
	@if [ -f projects/$@/scripts/generate_shelves_with_brackets.py ]; then \
		$(PYTHON) projects/$@/scripts/generate_shelves_with_brackets.py; \
	else echo "    (skipped — not applicable to this project)"; fi
	@echo "==> [$@] 3/5  per-sheet nesting layouts"
	@if [ -f projects/$@/scripts/generate_nested_layouts.py ]; then \
		$(PYTHON) projects/$@/scripts/generate_nested_layouts.py; \
	else echo "    (skipped — not applicable to this project)"; fi
	@echo "==> [$@] 4/5  export nesting_geometry.json"
	@if [ -f projects/$@/configs/stud_positions.json ]; then \
		$(PYTHON) tools/export_shelf_geometry.py --project $@; \
	else echo "    (skipped — no stud_positions.json)"; fi
	@echo "==> [$@] 5/5  CNC contour + pocket DXFs"
	@if [ -f projects/$@/scripts/process_for_cnc.py ]; then \
		$(PYTHON) projects/$@/scripts/process_for_cnc.py; \
	elif [ -f projects/$@/nesting_layout.json ] && [ -f projects/$@/configs/stud_positions.json ]; then \
		$(PYTHON) tools/process_for_cnc.py --project $@; \
	else echo "    (skipped — no nesting_layout.json / stud_positions.json)"; fi
	@echo "==> [$@] done"
