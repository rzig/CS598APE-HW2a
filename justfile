alias b := build
alias c := clean
alias ga := gui_assess
alias gd := gui_data
alias h := hotspot

gui_assess name : (do_profile name "assess_ext")
    @just gui_view
gui_data name : (do_profile name "data_access")
    @just gui_view

hotspot name: (do_perf name)
    hotspot --sourcePaths . --appPath . output/perf.data 

gui_view:
    #!/bin/sh
    AMDuProf --session /tmp/prof/$(ls /tmp/prof) --src-path $(realpath .) --bin-path $(realpath .)

tui_view:
    #!/bin/sh
    AMDuProfCLI report -i /tmp/prof/$(ls /tmp/prof) --src-path $(realpath .) --bin-path $(realpath .)
    vd /tmp/prof/$(ls /tmp/prof)/report.csv

do_perf name: (perf_profile "./genetic_benchmark " + name)
 
do_profile name mode: (profile mode "./genetic_benchmark " + name)

profile mode *args: build
    #!/bin/sh
    rm -rf /tmp/prof/*
    echo {{args}}
    AMDuProfCLI collect --config {{mode}} -o /tmp/prof {{args}}

perf_profile *args: build
    perf record -o output/perf.data -e cache-misses,cycles,page-faults --call-graph dwarf {{args}}
    
build:
    make -f OriginalMakefile all

clean:
    make -f OriginalMakefile clean
