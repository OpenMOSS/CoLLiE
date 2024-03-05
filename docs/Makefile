# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXAPIDOC  = sphinx-apidoc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = collie
SPHINXEXCLUDE =
SOURCEDIR     = source
BUILDDIR      = collie
PORT          = 8000

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf html source/_build

build:
	cd $(SOURCEDIR) && python -m sphinx -T -b html -d _build/doctrees -D language=zh . ../html

server:
	cd $(BUILDDIR)/html && python -m http.server $(PORT)

server-versions:
	cd $(BUILDDIR) && python -m http.server $(PORT)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)