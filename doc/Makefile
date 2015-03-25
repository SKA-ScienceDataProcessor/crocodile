
default: docset.zip

aa.html: aa.rstw ../synthesis.py
	PYTHONPATH=../:${PYTHONPATH} Pweave aa.rstw ; rst2html aa.rst > aa.html

docset.zip: aa.html
	zip -r $@ $^ figures
