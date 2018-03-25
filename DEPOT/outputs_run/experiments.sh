#!/bin/bash
# param = $(-a)
# pwd
# ls $param

#model = $("BOX")
#inputdatafile = $("input_data_file_002.dat")

#j = $(d)
#for i in {1..3}
#do
#    if($i = 1); then $i = "" $j = ""  
#  	echo "now run -e Tracer $j $i monolithic_pypofacets.py $mode $inputdatafile"
#done
#now run -e Tracer monolithic_pypofacets.py BOX input_data_file_002.dat
#now run -e Tracer -d 1 monolithic_pypofacets.py BOX input_data_file_002.dat
#now run -e Tracer -d 2 monolithic_pypofacets.py BOX input_data_file_002.dat

#now dataflow 1 | dot -T svg -o t1monodd.svg
#now dataflow 1 -d 1 | dot -T svg -o t1monodd1.svg
#now dataflow 1 -d 2 | dot -T svg -o t1monodd2.svg

#now dataflow 2 | dot -T svg -o t2monod1d.svg
#now dataflow 2 -d 1 | dot -T svg -o t2monod1d1.svg
#now dataflow 2 -d 2 | dot -T svg -o t2monod1d2.svg

#now dataflow 3 | dot -T svg -o t3monod2d.svg
#now dataflow 3 -d 1 | dot -T svg -o t3monod2d1.svg
#now dataflow 3 -d 2 | dot -T svg -o t3monod2d2.svg

now run -e Tracer modular_pypofacets.py BOX input_data_file_002.dat
now run -e Tracer -d 1 modular_pypofacets.py BOX input_data_file_002.dat
now run -e Tracer -d 2 modular_pypofacets.py BOX input_data_file_002.dat

now dataflow 4 | dot -T svg -o t4moduldd.svg
now dataflow 4 -d 1 | dot -T svg -o t4moduldd1.svg
now dataflow 4 -d 2 | dot -T svg -o t4moduldd2.svg

now dataflow 5 | dot -T svg -o t5moduld1d.svg
now dataflow 5 -d 1 | dot -T svg -o t5moduld1d1.svg
now dataflow 5 -d 2 | dot -T svg -o t5moduld1d2.svg

now dataflow 6 | dot -T svg -o t6moduld2d.svg
now dataflow 6 -d 1 | dot -T svg -o t6moduld2d1.svg
now dataflow 6 -d 2 | dot -T svg -o t6moduld2d2.svg
