#!/bin/bash

psnr ()
{
    echo "--> $f"
    ff_psnr=`$FFMPEG_PATH/ffmpeg -i $1 -i $2 -lavfi psnr -nostats -f null - 2>&1 | grep "PSNR"`
    ff_psnr=${ff_psnr:38}
    ff_psnr=`sed -E "s/(y|u|v|average|min|max)://g" <<< $ff_psnr`
    size=`stat --printf="%s" $2`
    echo "$size -- $ff_psnr"
    echo "$f $size $ff_psnr" >> $3
}

if [ $# -ne 2 ];
then
    echo "Usage: computePSNRs inputFile outputLogFile"
    exit 1
fi

rm $2

for f in *.ivf;
do
    psnr $1 $f $2
done
