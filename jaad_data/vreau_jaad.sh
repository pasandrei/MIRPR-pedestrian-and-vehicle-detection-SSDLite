#!/bash/bash
git clone https://github.com/ykotseruba/JAAD
cp *.py JAAD
curl http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip --output JAAD_clips.zip
echo "Unzip the clips"
