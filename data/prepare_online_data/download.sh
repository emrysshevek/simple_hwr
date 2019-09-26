read -p "IAM On-Line Handwriting DB username: " iam_username
wget http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/original-xml-all.tar.gz http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineImages-all.tar.gz http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/writers.xml http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz --user $iam_username --ask-password

# http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz - has xml stroke information divided by lines
# http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/original-xml-part.tar.gz - has xml stroke information by page; writer-ids; transcriptions
# http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/original-xml-all.tar.gz - same as above, but with some without ground truths / writer IDs


echo "Decompressing original xml"
mkdir original-xml-all
tar -zxf original-xml-all.tar.gz -C original-xml-all

echo "Decompressiong line level xml"
mkdir line-level-xml
tar -zxf lineStrokes-all.tar.gz -C line-level-xml

echo "Decompressing line images"
tar -zxf lineImages-all.tar.gz

echo "Removing downloaded tar.gz files"
rm -f original-xml-all.tar.gz
rm -f lineImages-all.tar.gz
rm -f lineStrokes-all.tar.gz

echo "Creating json file for model use..."
python3 prepare_online_data.py
