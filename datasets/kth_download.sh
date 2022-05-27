TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
  mkdir $TARGET_DIR/processed
  URL=http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz
  wget $URL -P $TARGET_DIR/processed
  tar -zxvf $TARGET_DIR/processed/kth.tar.gz -C $TARGET_DIR/processed/
  rm $TARGET_DIR/processed/kth.tar.gz

  mkdir $TARGET_DIR/raw
  for c in walking jogging running handwaving handclapping boxing
  do  
    URL=http://www.nada.kth.se/cvap/actions/"$c".zip
    wget $URL -P $TARGET_DIR/raw
    mkdir $TARGET_DIR/raw/$c
    unzip $TARGET_DIR/raw/"$c".zip -d $TARGET_DIR/raw/$c
    rm $TARGET_DIR/raw/"$c".zip
  done

fi