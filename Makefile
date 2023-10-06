all: build
.PHONY: install

install:
	@pip3 install .

uninstall:
	@pip3 uninstall -y infscale

reinstall: clean uninstall install

clean:
	@rm -rf build dist infscale.egg-info
