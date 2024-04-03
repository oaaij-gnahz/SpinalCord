#!/bin/bash
# path=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/DataFromBenPurple003/converted_data.mda
# outpath=/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/DataFromBenPurple003
# geom_file=channelmapBEN_notfinal.csv
path=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/MustangContinuous/Mustang_220126_125248/converted_data.mda
outpath=/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece
geom_file=map.csv
export ML_TEMPORARY_DIRECTORY=/media/hanlin/Liuyang_10T_backup/jiaaoZ/ml_temp
samplerate=30000
ovr_start_stamp=$SECONDS
if [ -d $outpath ] 
then
    echo "Directory \"$outpath\" exists." 
else
    echo "Creating directory: \"$outpath\""
    mkdir -p $outpath
fi
cat ./mountainSort32_cluster_only.sh > logs/Mustang220126ContinuousOnepiece_script.txt
echo ---------------------------------------------------------------------
echo Executing following command:
echo ./mountainSort32_cluster_only.sh $path $outpath $samplerate $geom_file
echo ---------------------------------------------------------------------
session_start_stamp=$SECONDS
./mountainSort32_cluster_only.sh $path $outpath $samplerate $geom_file
echo "Session finished. Deleting temp files..."
rm -rf $ML_TEMPORARY_DIRECTORY/*
echo "Session finished in " $(( SECONDS - session_start_stamp )) " seconds."
