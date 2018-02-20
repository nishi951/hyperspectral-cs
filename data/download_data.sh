# Harvard dataset

# Columbia cave dataset
CAVEDIR="CAVE"
if [ ! -d "$CAVEDIR" ]; then
    PREFIX=http://www.cs.columbia.edu/CAVE/databases/multispectral/zip
    wget ${PREFIX}/complete_ms_data.zip
    unzip complete_ms_data.zip -d $CAVEDIR
fi

# KAIST dataset
KAISTDIR="KAIST"
if [ ! -d "$KAISTDIR" ]; then
    mkdir $KAISTDIR
    cd $KAISTDIR
    PREFIX=http://vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset/exr
    for i in $(seq -f "%02g" 1 30); do
	wget ${PREFIX}/scene${i}_reflectance.exr
    done
    cd ..
fi
