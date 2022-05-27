# https://github.com/cemsaz/city-scapes-script
TARGET_DIR=$1
USERNAME=$2
PASSWORD=$3
if [ -z $TARGET_DIR ]
then
    echo "Must specify target directory"
else
    mkdir $TARGET_DIR/
    # Login
    wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$1&password=$2&submit=Login" https://www.cityscapes-dataset.com/login/ -P $TARGET_DIR
    # Download leftImg8bit_sequence_trainvaltest.zip (324GB)
    wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=14 -P $TARGET_DIR
    # Unzip
    unzip $TARGET_DIR/leftImg8bit_sequence_trainvaltest.zip -d $TARGET_DIR
fi
