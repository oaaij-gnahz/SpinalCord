#!/bin/bash

folders=( \
    "nora_221116_114205" \
    "nora_221117_225954" \
    "nora_221118_222513" \
    "nora_221122_185629" \
    "nora_221130_094643" \
    "nora_221204_204553" \
    "nora_221207_081815" \
    "nora_221213_224340" \
)
rootfolder=/media/G1/xl_spinal_cord_electrode/Animals/active_Animals/Nora_112522_7refOut/Recordings
destfolder=/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/nora_chronic

if [ ! -d $destfolder ]; then
  mkdir $destfolder
fi

for folder in "${folders[@]}"; do
  echo cp -rv $rootfolder/$folder $destfolder/$folder
  cp -rv $rootfolder/$folder $destfolder/$folder
done


