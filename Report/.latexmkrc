# LaTeX commands
$lualatex         = 'lualatex %O %S -shell-escape';
$pdflatex = 'pdflatex'.$texoption;
$latex_silent_switch = '-interaction=batchmode -c-style-errors';

# bibTeX commands
$bibtex    = 'upbibtex %O %B';
$biber     = 'biber %O --bblencoding=utf8 -u -U --output_safechars %B';
$makeindex = 'upmendex %O -o %D %S';

$dvipdf           = 'dvipdfmx %O -o %D %S';

# Typeset mode (generate a PDF)
$pdf_mode = 4;

# Other configuration
$pvc_view_file_via_temporary = 0;
$max_repeat = 5;
$clean_ext = "xmpdata";