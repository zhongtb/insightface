ls FDDB_pre_txt | while read line; 
do
        #echo "${line%_thresh*}"
        a=${line%_thresh*}
        echo "a"
        echo "${a}"

        #./evaluate -a ../ellipseList.txt -d ../dc_FDDB_pre_txt/${line}/results.txt -i ../originalPics/ -l ../imList.txt
        #cp tempDiscROC.txt ../dc_res/${line}_tempDiscROC.txt
done
