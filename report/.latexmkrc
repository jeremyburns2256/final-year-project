# latexmk configuration for the report.
#
# Send every generated file (main.pdf and the .aux/.log/.toc/... churn) to
# build/, so this directory holds only source. build/ is gitignored.
#
#   latexmk -pdf main.tex   ->  build/main.pdf
#   latexmk -C              ->  removes build/ contents

$pdf_mode = 1;          # build a PDF with pdflatex
$out_dir  = 'build';    # generated output, including main.pdf
$aux_dir  = 'build';    # .aux/.log/.toc and friends

# Create build/ (and any subdirectory latex asks for) if it is missing.
$allow_subdir_creation = 2;

# Let -C remove the generated .bbl too (the default keeps it), so a clean
# leaves no build output behind. references.bib is the source and is untouched.
$bibtex_use = 1.5;
