#!/bin/bash

if [ $# -ne 2 ];
then
    echo "Usage: genAV1 inputFolder outputDir"
    exit 1
fi


psnr ()
{
    ff_psnr=`ffmpeg -i $1 -i $2 -lavfi psnr -nostats -f null - 2>&1 | grep "PSNR"`
    ff_psnr=${ff_psnr:38}
    ff_psnr=`sed -E "s/(y|u|v|average|min|max)://g" <<< $ff_psnr`
    size=`stat --printf="%s" $2`
    echo "$size -- $ff_psnr"
    echo "$f $size $ff_psnr" >> $3
}


rm "output.log" "time.log"

for f in $1/*.y4m;
do
    echo "--> $f"
    filename=`basename $f .y4m`
    outputPath="$2/$filename.ivf"
    compTime=`{ time -p target/release/rav1e "$f" -o "$outputPath" -s 10; } 2>&1`

    compTimeReal=`echo "$compTime" | grep "real"`
    compTimeReal=`sed -E 's/(0m|s)//g;s/,/\./g' <<< $compTimeReal`
    compTimeReal=${compTimeReal:4}

    echo "$compTimeReal" >> "time.log"
    echo "$compTimeReal"
    psnr $f $outputPath "output.log"
done

