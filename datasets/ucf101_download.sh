TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
    echo "Must specify target directory"
else
    mkdir $TARGET_DIR/
    # Download UCF101.rar (6.5GB)
    wget -P $TARGET_DIR https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    # Unrar
    unrar x $TARGET_DIR/UCF101.rar $TARGET_DIR
    # Download splits
    wget -P $TARGET_DIR https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    # Unzip
    unzip $TARGET_DIR/UCF101TrainTestSplits-RecognitionTask.zip -d $TARGET_DIR
fi
