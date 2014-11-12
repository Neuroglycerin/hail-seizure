#!/bin/bash

for i in $(seq 2 8); do 
    sort -t$'\t' -k $i ${1} | awk -F '\t' '{print $1}' | head -n $2 >> intermediate_file;
done;


sort intermediate_file | uniq | awk '{ a[$0] } END {for (i in a) {for (j in a) {if (i != j)  print (i ", " j)}}}' >> combos_of_top_${2}_in_${1}

for i in {raw,ica,csp}; do
    sed -i "s/$i/cln,$i,dwn/g" combos_of_top_${2}_in_${1};
done

sed -i 's/SVC_//g' combos_of_top_${2}_in_${1}
sed -i 's/^/["/' combos_of_top_${2}_in_${1}
sed -i 's/$/_"]/' combos_of_top_${2}_in_${1}
sed -i 's/, /_", "/' combos_of_top_${2}_in_${1}
sed -i 's/dwn_/dwn_feat_/g' combos_of_top_${2}_in_${1}

rm intermediate_file
