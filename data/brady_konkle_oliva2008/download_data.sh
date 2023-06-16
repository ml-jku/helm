curl https://bradylab.ucsd.edu/stimuli/ObjectsAll.zip -o ObjectsAll.zip
unzip ObjectsAll.zip
pushd OBJECTSALL

python << EOM
import os
from PIL import Image
files = [f for f in os.listdir('.') if f.endswith('jpg') or f.endswith('JPG')]
for i, file in enumerate(sorted(files)):
  im = Image.open(file)
  im.save('../%04d.png' % (i+1))
EOM
popd
rm -rf __MACOSX OBJECTSALL ObjectsAll.zip

echo Dataset directory:
pwd
