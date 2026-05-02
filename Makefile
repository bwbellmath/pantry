PROJECTS := $(notdir $(wildcard projects/*))
PYTHON   ?= python3

.DEFAULT_GOAL := help
.PHONY: help $(PROJECTS)

help:
	@echo "Available projects: $(PROJECTS)"
	@echo "Usage: make <project_name>     # rebuild every output for that project"

$(PROJECTS):
	@echo "==> [$@] 1/5  generate shelf outlines"
	$(PYTHON) projects/$@/scripts/generate.py
	@echo "==> [$@] 2/5  overlay brackets (visualization DXF + PDF)"
	$(PYTHON) projects/$@/scripts/generate_shelves_with_brackets.py
	@echo "==> [$@] 3/5  per-sheet nesting layouts"
	$(PYTHON) projects/$@/scripts/generate_nested_layouts.py
	@echo "==> [$@] 4/5  export nesting_geometry.json"
	$(PYTHON) tools/export_shelf_geometry.py --project $@
	@echo "==> [$@] 5/5  CNC contour + pocket DXFs"
	$(PYTHON) tools/process_for_cnc.py --project $@
	@echo "==> [$@] done"
