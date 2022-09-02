all: create clean

create:
	pdflatex main-tx-soil-moisture.tex
	bibtex  main-tx-soil-moisture
	pdflatex main-tx-soil-moisture.tex


clean:
	rm -rf *.aux  *.nav  *.vrb  *.log  *.out *.snm *.toc *.gz *.mtc* *.maf *.fls *.tmp *.bbl *.blg  *.mtc *.fdb_latexmk
