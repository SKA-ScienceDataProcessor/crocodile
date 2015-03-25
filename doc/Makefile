PWs=   wimaging.rstw aa.rstw
HTMLs= $(PWs:.rstw=.html)

default: docset.zip

%.html: %.rstw ../synthesis.py
	PYTHONPATH=../:${PYTHONPATH} Pweave $< ; rst2html $(<:.rstw=.rst)  > $@

docset.zip: $(HTMLs)
	zip -r $@ $^ figures
