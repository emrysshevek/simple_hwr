wget http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/original-xml-all.tar.gz http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineImages-all.tar.gz http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/writers.xml --user nnmllab --password datasets

echo "Decompressing original xml"
mkdir original-xml-all
tar -zxf original-xml-all.tar.gz -C original-xml-all

echo "Decompressing line images"
tar -zxf lineImages-all.tar.gz

echo "Removing downloaded tar.gz files"
rm -f original-xml-all.tar.gz
rm -f lineImages-all.tar.gz

echo "Creating json file for model use..."
python3 prepare_online_data.py
