#!/bin/bash
animal_folder=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/MustangContinuous/Mustang_220126_125248
output_folder=/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/dt375Segged
geom_file=map.csv
# geom_file=map_corolla24ch.csv
#animal_folder=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/testdata
#output_folder=/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/testdata/outputs_thres3
#geom_file=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/testdata/geom.csv
export ML_TEMPORARY_DIRECTORY=/media/hanlin/Liuyang_10T_backup/jiaaoZ/ml_temp
if [ -d $output_folder ] 
then
    echo "Directory \"$output_folder\" exists." 
else
    echo "Creating directory: \"$output_folder\""
    mkdir -p $output_folder
fi
samplerate=30000
ovr_start_stamp=$SECONDS
mdas=$(ls $animal_folder)
# mdas=("converted_data_seg2")
i=1
for mdafile in ${mdas[*]}
  do
    path=$animal_folder/$mdafile
    outpath=$output_folder/seg$i
    ((i=i+1))
    if [ -d $outpath ] 
    then
        echo "Directory \"$outpath\" exists." 
    else
        echo "Creating directory: \"$outpath\""
        mkdir $outpath
    fi
    echo ---------------------------------------------------------------------
    echo Executing following command:
    echo ./mountainSort32_spinalCordGait_jz103.sh $path $outpath $samplerate $geom_file
    echo ---------------------------------------------------------------------
    session_start_stamp=$SECONDS
    ./mountainSort32_spinalCordGait_jz103.sh $path $outpath $samplerate $geom_file
    echo "Session finished. Deleting temp files..."
    rm -rf $ML_TEMPORARY_DIRECTORY/*
    echo "Session finished in " $(( SECONDS - session_start_stamp )) " seconds."
  done
echo "All sessions done in " $(( SECONDS - ovr_start_stamp )) " seconds."
